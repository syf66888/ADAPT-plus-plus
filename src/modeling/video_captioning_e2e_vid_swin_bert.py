import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
import cv2
import torchvision.transforms.functional as F


class VideoTransformer(torch.nn.Module):
    """ This is the one head module that performs Dirving Caption Generation. """

    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        """ Initializes the model.
        Parameters:
            args: basic args of ADAPT, mostly defined in `src/configs/VidSwinBert/BDDX_multi_default.json` and input args
            config: config of transformer_encoder, mostly defined in `models/captioning/bert-base-uncased/config.json`
            swin: torch module of the backbone to be used. See `src/modeling/load_swin.py`
            transformer_encoder: torch module of the transformer architecture. See `src/modeling/load_bert.py`
        """
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size + 1, self.img_feature_dim)
        self.compute_mask_on_the_fly = False  # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        self.max_num_frames = getattr(args, 'max_num_frames', 2)
        self.expand_car_info = torch.nn.Linear(self.max_num_frames, self.img_feature_dim)

        # add sensor information
        self.use_car_sensor = getattr(args, 'use_car_sensor', False)

        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)

        if self.learn_mask_enabled == True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length * args.max_img_seq_length, 1)
            self.sigmoid = torch.nn.Sigmoid()

    def compute_optical_flow(self, images):
        optical_flow_feats = []
        if torch.cuda.is_available():
            device = torch.device("cuda")
        B, C, S, H, W = images.shape  # batch, channel, segment, height, width
        images = (images * 255).to(torch.uint8).clamp(0, 255)
        data_type = images.dtype
        for b in range(B):  # Iterate over batches
            optical_flow_batch = []
            for i in range(S - 1):  # Iterate over frames
                images1 = images[b, :, i].cpu()
                images2 = images[b, :, i + 1].cpu()
                prev_frame = cv2.cvtColor(images1.permute(1, 2, 0).numpy(),
                                          cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                next_frame = cv2.cvtColor(images2.permute(1, 2, 0).numpy(),
                                          cv2.COLOR_RGB2GRAY)  # Convert to grayscale

                prev_points = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01,
                                                      minDistance=30)
                if prev_points is None or prev_points.size == 0:
                    x_flow_mean = 0.0
                    optical_flow_feats = torch.zeros((B, 31, 1))
                    optical_flow_feats = optical_flow_feats.to(device)
                    return optical_flow_feats

                try:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_points, None)
                    good_prev_points = prev_points[status == 1]
                    good_next_points = next_points[status == 1]
                    flow_vectors = good_next_points - good_prev_points
                    x_flow = flow_vectors[:, 0]
                    if x_flow.size > 0:
                        x_flow_mean = x_flow.mean()
                    else:
                        x_flow_mean = 0.0
                    optical_flow_batch.append(x_flow_mean)
                except:
                    optical_flow_feats = torch.zeros((B, 31, 1))
                    optical_flow_feats = optical_flow_feats.to(device)
                    return optical_flow_feats

            optical_flow_feats.append(optical_flow_batch)

        optical_flow_feats = torch.tensor(optical_flow_feats)
        optical_flow_feats = optical_flow_feats.unsqueeze(-1)
        optical_flow_feats = optical_flow_feats.to(device)
        return optical_flow_feats

    def forward(self, *args, **kwargs):
        """ The forward process of ADAPT,
        Parameters:
            input_ids: word tokens of input sentences tokenized by tokenizer
            attention_mask: multimodal attention mask in Vision-Language transformer
            token_type_ids: typen tokens of input sentences,
                            0 means it is a narration sentence and 1 means a reasoning sentence, same size with input_ids
            img_feats: preprocessed frames of the video
            masked_pos: [MASK] position when performing MLM, used to locate the masked words
            masked_ids: groung truth of [MASK] when performing MLM
        """
        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args = ()

        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        # 计算稀疏光流特征
        optical_flow_feats = self.compute_optical_flow(images)
        # todo：在这里输出optical_flow_feats.shape看看形状
        # 不用看了，生成的形状是(B x S-1 x 2) ,需要打印看看B是多少
        vid_feats = self.swin(images)

        # tokenize video features to video tokens
        if self.use_grid_feat == True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        optical_flow_feats = optical_flow_feats.to(vid_feats.dtype)
        B1, M, latent = vid_feats.shape
        padded_feats = torch.zeros(B, M, 1, device=optical_flow_feats.device, dtype=optical_flow_feats.dtype)
        padded_feats[:, :31, :] = optical_flow_feats
        fused_feats = torch.cat((vid_feats, padded_feats), dim=2)

        # 得到的维度是 (B, C x S x H x W //latent_feat_size , latent_feat_size)

        # use an mlp to transform video token dimension
        vid_feats = self.fc(fused_feats)

        # 融合特征，将光流特征与swin transformer提取的特征融合输入到模型当中去
        # todo：在这里加一个全连接层转换到二维
        # fused_feats = torch.cat((vid_feats, optical_flow_feats), dim=1)
        # todo：将vid_feats与optical_flow_feats拼接起来（这个时候是二维的拼接）
        # todo：再将拼接后的fused_feats输入到一个全连接层当中去将其转为bert能够接受的输入类型

        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats

        # disable bert attention outputs to avoid some bugs
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)

        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask) * learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att >= 0.5) * 1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

        # Driving Caption Generation head
        outputs = self.trans_encoder(*args, **kwargs)

        # sparse attention mask loss
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)
            outputs = outputs + (loss_sparsity,)

        return outputs

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad = not freeze

