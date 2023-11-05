
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

#######################
# class GaussianNoiseLayer(nn.Module):
#     def __init__(self, std):
#         super(GaussianNoiseLayer, self).__init__()
#         self.std = std

#     def forward(self, input):
#         if self.training:
#             noise = torch.randn(input.size(), dtype=input.dtype, device=input.device) * self.std
#             return input + noise
#         return input
    
# class InstanceNorm(nn.Module):
#     def __init__(self, input_dim):
#         super(InstanceNorm, self).__init__()
        
#     def forward(self, input):
#         depth = input.size(1)
#         with torch.no_grad():
#             self.scale = nn.Parameter(torch.randn(input.shape, dtype=torch.float32)).cuda()
#             self.offset = nn.Parameter(torch.zeros(input.shape, dtype=torch.float32)).cuda()
#             mean = torch.mean(input, dim=[2, 3], keepdim=True)
#             variance = torch.var(input, dim=[2, 3], keepdim=True)
#             epsilon = 1e-5
#             inv = 1.0 / torch.sqrt(variance + epsilon)
#             normalized = (input - mean) * inv
#             # print(normalized.shape)
#             # print(self.scale.shape)
#             # print(self.offset.shape)
        
#         return self.scale * normalized + self.offset

# def conv2d(input_dim, output_dim, ks=4, s=2, stddev=0.02):
#     conv_layer = nn.Conv2d(input_dim, output_dim, ks, s, padding=ks // 2)
#     conv_layer.weight.data.normal_(0.0, stddev)
#     return conv_layer

# def deconv2d(input_dim, output_dim, ks=4, s=2, stddev=0.02):
#     deconv_layer = nn.ConvTranspose2d(input_dim, output_dim, ks, s, padding=ks // 2)
#     deconv_layer.weight.data.normal_(0.0, stddev)
#     return deconv_layer

# def gaussian_noise_layer(input_layer, std):
#     noise = torch.randn(input_layer.size(), dtype=torch.float32) * std
#     return input_layer + noise.cuda()

# class ResidualBlock(nn.Module):
#     def __init__(self, dim, ks=3, s=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, ks, s, padding=ks // 2)
#         self.conv2 = nn.Conv2d(dim, dim, ks, s, padding=ks // 2)
#         self.norm1 = InstanceNorm(dim)
#         self.norm2 = InstanceNorm(dim)

#     def forward(self, x):
#         y = self.norm1(self.conv1(x))
#         y = F.relu(y)
#         y = self.norm2(self.conv2(y))
#         return y + x

# class GeneratorResNet(nn.Module):
#     def __init__(self, options):
#         super(GeneratorResNet, self).__init__()
#         self.options = options
#         self.encoder = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             conv2d(options.input_c_dim, options.gf_dim, 7, 1),
#             InstanceNorm(options.gf_dim),
#             nn.ReLU(),
#             conv2d(options.gf_dim, options.gf_dim * 2, 3, 2),
#             InstanceNorm(options.gf_dim*2),
#             nn.ReLU(),
#             conv2d(options.gf_dim * 2, options.gf_dim * 4, 3, 2),
#             InstanceNorm(options.gf_dim*4),
#             nn.ReLU()
#         )
#         self.residual_blocks = nn.ModuleList([ResidualBlock(options.gf_dim * 4) for _ in range(5)])
#         self.translation_decoder = nn.Sequential(
#             ResidualBlock(options.gf_dim * 4),
#             ResidualBlock(options.gf_dim * 4),
#             ResidualBlock(options.gf_dim * 4),
#             ResidualBlock(options.gf_dim * 4),
#             deconv2d(options.gf_dim * 4, options.gf_dim * 2, 3, 2),
#             InstanceNorm(options.gf_dim*2),
#             nn.ReLU(),
#             deconv2d(options.gf_dim * 2, options.gf_dim, 3, 2),
#             InstanceNorm(options.gf_dim),
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(options.gf_dim, options.output_c_dim, 7, 1),
#             nn.Tanh()
#         )
#         # self.reconstruction_decoder = nn.Sequential(
#         #     GaussianNoiseLayer(0.02),
#         #     ResidualBlock(options.gf_dim * 4),
#         #     GaussianNoiseLayer(0.02),
#         #     ResidualBlock(options.gf_dim * 4),
#         #     ResidualBlock(options.gf_dim * 4),
#         #     ResidualBlock(options.gf_dim * 4),
#         #     deconv2d(options.gf_dim * 4, options.gf_dim * 2, 3, 2),
#         #     InstanceNorm(options.gf_dim*2),
#         #     nn.ReLU(),
#         #     deconv2d(options.gf_dim * 2, options.gf_dim, 3, 2),
#         #     InstanceNorm(options.gf_dim),
#         #     nn.ReflectionPad2d(3),
#         #     nn.Conv2d(options.gf_dim, options.output_c_dim, 7, 1),
#         #     nn.Tanh()
#         # )

#     def forward(self, image):
#         c0 = F.pad(image, (3, 3, 3, 3), "reflect")
#         c1 = self.encoder(c0)
#         r = c1
#         for block in self.residual_blocks:
#             r = block(r)
#         r5 = r
#         r6 = self.translation_decoder(r5)
#         #r5 = gaussian_noise_layer(r5, 0.02)
#         # r6_rec = self.reconstruction_decoder(r5)
#         return r6

class DomainAgnosticClassifier(nn.Module):
    def __init__(self, options):
        super(DomainAgnosticClassifier, self).__init__()

        self.conv1 = conv2d(3, options.df_dim * 4)
        self.norm1 = InstanceNorm(options.df_dim * 4)
        self.conv2 = conv2d(options.df_dim * 4, options.df_dim * 2)
        self.norm2 = InstanceNorm(options.df_dim * 2)
        self.conv3 = conv2d(options.df_dim * 2, options.df_dim * 2)
        self.norm3 = InstanceNorm(options.df_dim * 2)
        self.prediction = conv2d(options.df_dim * 2, 2)

    def forward(self, percep):
        h1 = F.leaky_relu(self.norm1(self.conv1(percep)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = self.prediction(h3)
        h4_mean = h4.mean(dim=(0, 2, 3), keepdim=True)
        return h4_mean.view(-1, 1, 1, 2)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_block):
        super(GeneratorResNet, self).__init__()
        
        channels = input_shape[0]
        
        # Initial Convolution Block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2), # --> width*2, heigh*2
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output Layer
        model += [nn.ReflectionPad2d(channels),
                  nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()
                 ]
        
        # Unpacking
        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        
        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height//2**4, width//2**4)
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
    def forward(self, img):
        return self.model(img)
############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,

        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, proposal_index = None
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1


        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "consistency_target":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            proposals_roih, proposals_into_roih, proposal_index = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=gt_instances,
                compute_loss=False,
                branch=branch,
            )

            return proposal_losses,proposals_into_roih, proposals_rpn, proposals_roih, proposal_index

        elif branch == "unsup_data_consistency":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            # print("Images: " + str(images))
            # print("Features: " + str(features))
            # print("given_proposals: " + str(given_proposals))
        
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                given_proposals,
                targets=None,
                compute_loss=False,
                branch=branch,
                proposal_index=proposal_index,
            )

            return {}, [], proposals_roih, ROI_predictions
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


