import os
import time
import logging
import json
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
import torchvision.transforms as T
from torch.nn import functional as F
from FDA.utils import FDA_source_to_target_unet, unet_helper
import torchvision.transforms as transforms

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
import segmentation_models_pytorch as smp
from twophase.modeling.meta_arch.rcnn import FCDiscriminator_img
from twophase.modeling.meta_arch.rcnn import Discriminator



from twophase.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from twophase.data.dataset_mapper import DatasetMapperTwoCropSeparate
from twophase.engine.hooks import LossEvalHook
from twophase.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from twophase.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from twophase.solver.build import build_lr_scheduler
from twophase.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from twophase.modeling.custom_losses import ConsistencyLosses
from twophase.data.transforms.night_aug import NightAug
from torchvision.utils import save_image
import copy

class L_TV(nn.Module):
    def __init__(self, mid_val=None):
        super(L_TV,self).__init__()
        self.mid_val = mid_val

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        if self.mid_val is None:
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        else:
            h_tv = torch.pow(torch.clamp(self.mid_val - torch.abs(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]) - self.mid_val), 0, 1), 2).sum()
            w_tv = torch.pow(torch.clamp(self.mid_val - torch.abs(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]) - self.mid_val), 0, 1), 2).sum()
        return 2 * (h_tv/count_h+w_tv/count_w)/batch_size

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k
                
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch+self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# Adaptive Teacher Trainer
class TwoPCTrainer(DefaultTrainer):
    
    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
                torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)
                
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        
        self.unet_model = smp.Unet('resnet101', encoder_weights='imagenet', classes=6, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).cuda()
        unet_checkpoint_path = './output/bdd100k_unet/Unet_NightDA25000.pth'
        unet_checkpoint = torch.load(unet_checkpoint_path)
        self.unet_model.load_state_dict(unet_checkpoint)
        max_lr = 0.010#1e-3
        epoch = 15
        weight_decay = 1e-4#1e-4
        #self.optimizer_unet = torch.optim.AdamW(self.unet_model.parameters(), lr=max_lr, weight_decay=weight_decay)
        input_shape = (3, 600, 1067)
        # self.discriminator = Discriminator(input_shape).cuda()#FCDiscriminator_img(num_classes=1067*600).cuda()
        # disc_checkpoint_path = 'output/bdd100k_unet/Disc_NightDA5000.pth'
        # disc_checkpoint = torch.load(disc_checkpoint_path)
        # self.discriminator.load_state_dict(disc_checkpoint)
        #self.discriminator.apply(self.weights_init_normal)
        # training
        epoch = 0 # epoch to start training from
        n_epochs = 5 # number of epochs of training
        batch_size = 1 # size of the batches
        lr = 0.0002 # adam : learning rate
        b1 = 0.5 # adam : decay of first order momentum of gradient
        b2 = 0.999 # adam : decay of first order momentum of gradient
        decay_epoch = 3 # suggested default : 100 (suggested 'n_epochs' is 200)
        #self.optimizer_D_A = torch.optim.Adam(
        #self.discriminator.parameters(), lr=0.0001, betas=(b1,b2))#0.0002
        # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        #                     self.optimizer_D_A,
        #                     lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        #                 )
        

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)


        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.stu_scale = None
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.scale_list = np.array(cfg.SEMISUPNET.SCALE_LIST)
        self.scale_checkpoints = np.array(cfg.SEMISUPNET.SCALE_STEPS)
        self.cfg = cfg
        self.ext_data = []
        self.img_vals = {}
        self.consistency_losses = ConsistencyLosses()
        self.night_aug = NightAug()

        self.register_hooks(self.build_hooks()) 

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume: # and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        # if hasattr(self, "_last_eval_results") and comm.is_main_process():
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                self.unet_model.train()
                #self.discriminator.train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    def gradient_magnitude(self, img):

        # Compute gradients in x and y directions
        gradient_x = F.conv2d(img, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, requires_grad=True).cuda().unsqueeze(0).unsqueeze(0), padding=1).cuda()
        gradient_y = F.conv2d(img, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, requires_grad=True).cuda().unsqueeze(0).unsqueeze(0), padding=1).cuda()
        
        #print("gradient_x: "+str(gradient_x.requires_grad))
        #print("gradient_y: "+str(gradient_y.requires_grad))
        
        # Compute gradient magnitude
        criterion_mse = torch.nn.MSELoss().cuda()
        grad_mag = criterion_mse(gradient_x, gradient_y).cuda()
        #print("grad_mag: "+str(grad_mag.requires_grad))
        
        return grad_mag.cuda()
    

    def edge_loss(self, gen_image, ground_truth):
        # Compute gradient magnitude for generated and ground truth images
        gen_grad_mag = self.gradient_magnitude(gen_image)
        ground_truth_grad_mag = self.gradient_magnitude(ground_truth)
        
        # Compute the difference in gradient magnitudes
        edge_loss = F.mse_loss(gen_grad_mag, ground_truth_grad_mag).cuda()
        
        return edge_loss
    
    def compute_histogram(image, num_bins=256):
        """
        Compute histogram of an image.

        image: Input image (assumed to be grayscale, single channel)
        num_bins: Number of bins for the histogram
        """
        # Flatten the image and compute the histogram
        histogram = torch.zeros(num_bins, dtype=torch.float32)
        flattened_image = image.view(-1)
        for i in range(num_bins):
            histogram[i] = torch.sum(flattened_image == i)
        return histogram / torch.sum(histogram)  # Normalize the histogram

    def bhattacharyya_distance(self, img1, img2):
        """
        Compute Bhattacharyya Distance between two histograms.

        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
        """
        # hist1 = self.compute_histogram(img1)
        # hist2 = self.compute_histogram(img2)
        # print("b_distance: " + str(b_distance))
        # # Bhattacharyya Distance
        # b_distance = -torch.log(torch.sum(torch.sqrt(hist1 * hist2))).cuda()
        criterion_cycle = torch.nn.L1Loss().cuda()
        cycle_loss = criterion_cycle(img1, img2)
        print("cycle_loss: " + str(cycle_loss))
        
        return cycle_loss.cuda() #+ b_distance.cuda() 
    
    def color_regularization_loss(self, fake_image, org_img):
        # Calculate mean color for each channel (assuming image shape is [batch, channels, height, width])
        mean_color_fake = torch.mean(fake_image, dim=(2, 3)).cuda()  # Compute mean color for each channel
        mean_color_org = torch.mean(org_img, dim=(2, 3)).cuda()  # Compute mean color for each channel
        

        # Compute loss as the sum of squared deviations from the mean color
        loss = torch.sum((mean_color_fake-mean_color_org) ** 2).cuda()

        return loss
    

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def print_gradients(self, model):
        for name, param in model.named_parameters():
            print(f'{name}: {param.grad}: {param.requires_grad}')

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        start_iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data, unlabel_data = data
        data_time = time.perf_counter() - start
        
        # criterion_GAN = torch.nn.MSELoss().cuda()
        # criterion_cycle = torch.nn.L1Loss().cuda()
        # criterion_identity = torch.nn.L1Loss()
        # loss_color = L_color().cuda()
        # loss_variation = L_TV(mid_val=0.02).cuda()
        # source_label = 0
        # target_label = 1
        # cuda = torch.cuda.is_available()
        # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        # valid = Tensor(np.ones((1, *self.discriminator.output_shape))) # requires_grad = False. Default.
        # #print("Valid: " + str(valid.shape))
        # fake = Tensor(np.zeros((1, *self.discriminator.output_shape))) # requires_grad = False. Default.
        # # label_images = [] #label_data[0]["image"].size(0)
        # # unlabel_images = []
        # # for item, item2 in zip(label_data, unlabel_data):
        # #     label_images.append(torch.tensor(item["image"]))
        # #     unlabel_images.append(torch.tensor(item2["image"]))

        # # # Stack the tensors
        # # label_tensor = torch.stack(label_images)
        # # unlabel_tensor = torch.stack(unlabel_images)

        
        # torch.autograd.set_detect_anomaly(True)
        # record_dict = {}
        # #print("Before unet weights" + str(self.unet_model.state_dict()), file=open('before_unet.txt', 'a'))
        # # self.print_gradients(self.unet_model)
        
        # # self.optimizer_unet.zero_grad()
        # # # self.print_gradients(self.unet_model)
        # # transform = transforms.Pad([10, 4, 11, 4])

        # # # loss_target_consistency = torch.sum(torch.stack(
        # # #     [self.bhattacharyya_distance((unet_helper(
        # # #             self.unet_model(
        # # #                 torch.unsqueeze(
        # # #                     transform((unlabel_img["image"].float()/255.0).cuda()), 0)
        # # #                 ))*255).type(torch.uint8), 
        # # #         (unlabel_img["image"].float()).cuda()) for unlabel_img in unlabel_data]))*5
        # # loss_target_consistency = torch.sum(torch.stack(
        # #     [criterion_cycle(unet_helper(
        # #             self.unet_model(
        # #                 torch.unsqueeze(
        # #                     transform((unlabel_img["image"].float()/255.0).cuda()), 0)
        # #                 )), 
        # #         (unlabel_img["image"].float()/255.0).cuda()) for unlabel_img in unlabel_data]))*5
        
        # # loss_target_consistency.backward()
        
        # # self.optimizer_unet.step()
        
        # self.optimizer_unet.zero_grad()
        # # self.print_gradients(self.unet_model)
        
        # # Print the weights of the first layer
        # # print(self.unet_model[0].weight)

        # # # Print the gradients of the first layer
        # # print(self.unet_model[0].weight.grad)
        # src_in_trg = [FDA_source_to_target_unet(x["image"], y["image"], self.unet_model, start_iter) for x, y in zip(label_data, unlabel_data)]
        # #print(str(start_iter)+" transformation: " + str(src_in_trg))
        # image_consistency_lambda = 3
        # image_darkness_lambda = 2
        
        # if start_iter%50 == 0:
        #     for x, y in zip(label_data, src_in_trg):
        #         avg_img = torch.mean(torch.stack((((x["image"]/255)*0.25).unsqueeze(0).cuda(), (y["src_org"]).unsqueeze(0))), dim=0)
        #         avg_img_tensor = avg_img[:, [2, 1, 0], :, :]
        #         rgb_img_tensor = (x["image"]/255).unsqueeze(0)
        #         rgb_img_tensor = rgb_img_tensor[:, [2, 1, 0], :, :] 
        #         rgb_img_tensor_2 = (y["image"]).unsqueeze(0)
        #         rgb_img_tensor_2 = rgb_img_tensor_2[:, [2, 1, 0], :, :] 
        #         rgb_img_tensor_3 = (y["fake_amp"]).unsqueeze(0)
        #         rgb_img_tensor_3 = rgb_img_tensor_3[:, [2, 1, 0], :, :] #y["trg_amp"]
        #         rgb_img_tensor_4 = (y["src_pha_org"]).unsqueeze(0)
        #         rgb_img_tensor_4 = rgb_img_tensor_4[:, [2, 1, 0], :, :] #y["src_org"]
        #         rgb_img_tensor_5 = (y["D_gen"]).unsqueeze(0)
        #         rgb_img_tensor_5 = rgb_img_tensor_5[:, [2, 1, 0], :, :] #y["src_org"]
        #         rgb_img_tensor_6 = (y["src_org"]).unsqueeze(0)
        #         rgb_img_tensor_6 = rgb_img_tensor_6[:, [2, 1, 0], :, :]#trg_img
        #         rgb_img_tensor_7 = (y["trg_img"]).unsqueeze(0)
        #         rgb_img_tensor_7 = rgb_img_tensor_7[:, [2, 1, 0], :, :]
        #         save_image([rgb_img_tensor.squeeze().type(torch.float32).cuda(),rgb_img_tensor_2.squeeze().type(torch.float32).cuda(), avg_img_tensor.squeeze().type(torch.float32).cuda(), rgb_img_tensor_6.squeeze().type(torch.float32).cuda(), rgb_img_tensor_4.squeeze().type(torch.float32).cuda(), rgb_img_tensor_5.squeeze().type(torch.float32).cuda(), rgb_img_tensor_7.squeeze().type(torch.float32).cuda()], './demo_images/'+"src_"+str(start_iter)+'.png')
        #     #save_image([(label["image"]/255).type(torch.float32) for label in label_data],'./demo_images/'+"src_"+str(start_iter)+'.png')
        #     #save_image([label["image"].type(torch.float32) for label in src_in_trg],'./demo_images/'+"src_to_target_"+str(start_iter)+'.png')
        # loss_darkness = (torch.sum(torch.stack([criterion_cycle(x["src_org"], torch.zeros_like(x["image"])).cuda() for x in src_in_trg]).cuda())).cuda()
        # #loss_darkness = (torch.sum(torch.stack([criterion_cycle(x["src_org"], (y["image"]/255).cuda()).cuda() for x,y in zip(src_in_trg, unlabel_data)]).cuda())).cuda()
        
        # # loss_darkness = torch.sum(torch.stack(
        # #     [self.darkness_loss(
        # #         torch.unsqueeze((x["image"]/255).float(),0).cuda(), torch.unsqueeze(y["src_org"],0)
        # #         ) for x, y in zip(label_data, src_in_trg)]
        # #     ).cuda()).cuda()
        # #Consistency Loss
        # #loss_consistency_fda = torch.sum(torch.stack([criterion_cycle(x["src_org"].cuda(), (y["image"]/255).cuda()) for x, y in zip(src_in_trg, label_data)]).cuda()).cuda()
        # # print("src_org: "+str(loss_consistency_fda.requires_grad))
        
        # #loss_color_reg = torch.sum(torch.stack([self.color_regularization_loss(torch.unsqueeze(x["src_org"],0), torch.unsqueeze(y["image"]/255,0)) for x, y in zip(src_in_trg, unlabel_data)]).cuda()).cuda()
        
        # # loss_consistency_phase = torch.sum(
        # #     torch.stack([criterion_cycle(x["pha_fake"].cuda(), (x["pha_src"]).cuda()) for x in src_in_trg])).cuda()
        # # print("loss_consistency_phase: "+str(loss_consistency_phase.requires_grad))
        
        # #loss_consistency_amp = (torch.sum(torch.stack([criterion_cycle(x["fake_amp"].cuda(), (x["trg_amp"]).cuda()) for x in src_in_trg]))*1000).cuda()
        # #print("loss_consistency_phase: "+str(loss_consistency_amp.requires_grad))
        
        # loss_color_consistency = torch.sum(torch.stack([torch.mean(loss_color(torch.unsqueeze(x["image"], 0))).cuda() for x in src_in_trg]))*25
        # loss_variation_org = torch.sum(torch.stack([torch.mean(loss_variation(torch.unsqueeze(x["src_org"], 0))).cuda() for x in src_in_trg]))*160
        
        # #loss_consistency_fda_amp = torch.sum(torch.tensor([criterion_cycle(x["src_amp"], x["fake_amp"]) for x in src_in_trg], requires_grad=True)).cuda()
        # discriminator_img_out_t_faked = [self.discriminator(torch.unsqueeze(x["image"], 0).cuda()) for x in src_in_trg]#[self.discriminator(x) for x in src_in_trg]
        # #discriminator_img_out_t_faked_pha = [self.discriminator(torch.unsqueeze(x["src_pha_org"], 0).cuda()) for x in src_in_trg]#[self.discriminator(x) for x in src_in_trg]
        
        
        # # if loss_target_consistency<0.33:
        # #     discriminator_img_out_t_faked = [self.discriminator(torch.unsqueeze(x["src_org"], 0).cuda()) for x in src_in_trg]#[self.discriminator(x) for x in src_in_trg]
        # # else:
        # #     discriminator_img_out_t_faked = [self.discriminator(torch.unsqueeze(x["image"], 0).cuda()) for x in src_in_trg]#[self.discriminator(x) for x in src_in_trg]
            
        # loss_discriminator_t_faked = torch.sum(torch.stack([criterion_GAN(z, valid).cuda() for z in discriminator_img_out_t_faked]))
        # #loss_discriminator_t_faked_pha = torch.sum(torch.stack([criterion_GAN(z, valid).cuda() for z in discriminator_img_out_t_faked_pha]))
        
        # # Calculate the edge maps of the input and target images.
        # # Calculate the mean squared error between the edge maps of the input and target images.
        # loss_edge = torch.sum(torch.stack(
        #     [self.edge_loss(
        #         torch.unsqueeze(torch.mean((input_image["image"]/255).type(torch.float32).cuda(), dim=0, keepdim=True),0), 
        #         torch.unsqueeze(torch.mean(output_image["image"].type(torch.float32).cuda(), dim=0, keepdim=True),0)) 
        #      for input_image, output_image in zip(label_data, src_in_trg)]))*90
        
        # # print("loss_discriminator_t_faked: "+str(loss_discriminator_t_faked.requires_grad))
        # loss_gen = (loss_discriminator_t_faked)+(loss_darkness*loss_discriminator_t_faked)+loss_variation_org+loss_color_consistency+loss_edge#+(loss_edge)#+loss_color_reg+loss_color_consistency#+(loss_discriminator_t_faked_pha/5)#+loss_variation_org##+(loss_edge)#+loss_color_org+loss_variation_org#+loss_consistency_amp#+loss_consistency_fda#+loss_consistency_phase
        # #loss_darkness.backward()
        # #loss_consistency_fda.backward()
        # #loss_consistency_fda_amp.backward()
        # # Backpropagate through the graph
        # loss_gen.backward()
        # self.optimizer_unet.step()
        
        
        # #print("After unet weights" + str(self.unet_model.state_dict()), file=open('after_unet.txt', 'a'))
        # # Print the weights of the first layer
        # # print(self.unet_model[0].weight)

        # # # Print the gradients of the first layer
        # # print(self.unet_model[0].weight.grad)
        
        # # print("Before disc weights" + str(self.discriminator.state_dict()), file=open('before_disc.txt', 'a'))
        # # self.print_gradients(self.unet_model)
        # self.optimizer_D_A.zero_grad()
        # # self.print_gradients(self.unet_model)
        # #Discriminator Loss, fake = src, valid = target
        # self.unet_model.requires_grad_ = False
        # #loss_discriminator_t = 0
        
        # # if loss_discriminator_t_faked < 1:
        # #     discriminator_img_out_t = [self.discriminator(torch.unsqueeze(x["src_org"].detach(), 0).cuda()) for x in src_in_trg]#[self.discriminator(x) for x in src_in_trg]
        # #     loss_discriminator_t = torch.sum(torch.stack([criterion_GAN(z, fake).cuda() for z in discriminator_img_out_t]))
        # #     loss_discriminator_t.backward(retain_graph=True)
        # # else:
        # #     loss_discriminator_t = 0
        
        # discriminator_img_out_s = [self.discriminator(torch.unsqueeze(x["image"].float(), 0).cuda()) for x in label_data]#[self.discriminator(x) for x in src_in_trg]
        # loss_discriminator_s = torch.sum(torch.stack([criterion_GAN(z, fake).cuda() for z in discriminator_img_out_s]))
        # loss_discriminator_s.backward(retain_graph=True)
        
        
        # discriminator_img_out_real_t = [self.discriminator(torch.unsqueeze(x["image"].float(), 0).cuda()) for x in unlabel_data]#[self.discriminator(x) for x in src_in_trg]
        
        # # print("loss_discriminator_t: "+str(loss_discriminator_t.requires_grad))

        # #loss_discriminator_s = torch.sum(torch.tensor([F.binary_cross_entropy_with_logits(z, torch.FloatTensor(z.data.size()).fill_(source_label).cuda()).cuda() for z in discriminator_img_out_s], requires_grad=True)).cuda()
        # loss_discriminator_real_t = torch.sum(torch.stack([criterion_GAN(z, valid).cuda() for z in discriminator_img_out_real_t]))
        
        # #loss_discriminator_real_t = torch.sum(torch.stack([criterion_GAN(z, valid).cuda() for z in discriminator_img_out_real_t]))
        # # print("loss_discriminator_real_t: "+str(loss_discriminator_real_t.requires_grad))
        
        # #loss_disc = (loss_discriminator_t + loss_discriminator_s + loss_discriminator_real_t)/3
        # #loss_disc = (loss_discriminator_t  + loss_discriminator_real_t)/2
        
        # # loss_discriminator_t.backward()
        # # loss_discriminator_s.backward()
        # # loss_discriminator_real_t.backward()
    
        
        # loss_discriminator_real_t.backward()
        # # print("After disc weights" + str(self.discriminator.state_dict()), file=open('after_disc.txt', 'a'))
        # self.optimizer_D_A.step()
        # # print("After disc weights" + str(self.discriminator.state_dict()), file=open('after_disc.txt', 'a'))
        
        
        # temp_dict = {
        #         'loss_darkness': loss_darkness,
        #         #'loss_color_reg': loss_color_reg,
        #         #'loss_target_consistency': loss_target_consistency,
        #         'loss_color_consistency': loss_color_consistency,
        #         'loss_variation_org': loss_variation_org,
        #         'loss_edge': loss_edge,
        #         #'loss_consistency_fda': loss_consistency_fda,
        #         # 'loss_consistency_fda_amp': loss_consistency_fda_amp,
        #         #'loss_consistency_amp': loss_consistency_amp,
        #         #'loss_consistency_phase': loss_consistency_phase,
        #         'loss_discriminator_t_faked': loss_discriminator_t_faked,
        #         #'loss_discriminator_t_faked_pha': loss_discriminator_t_faked_pha,
        #         #'loss_disc': loss_disc,
        #         #'loss_discriminator_gen_t': loss_discriminator_t,
        #         'loss_discriminator_s': loss_discriminator_s,
        #         'loss_discriminator_real_t': loss_discriminator_real_t,
        #     }
        # #print(temp_dict)
        # record_dict.update(temp_dict)
        
        # if (start_iter%5000 == 0):
        #     torch.save(self.unet_model.state_dict(
        #             ), 'output/bdd100k_unet/' + "/Unet_NightDA" + str(start_iter) + '.pth')
        #     torch.save(self.discriminator.state_dict(
        #             ), 'output/bdd100k_unet/' + "/Disc_NightDA" + str(start_iter) + '.pth')

        # Add NightAug images into supervised batch
        if self.cfg.NIGHTAUG and start_iter > 5000:
            #label_data_aug = self.night_aug.aug([x.copy() for x in label_data])
            with torch.no_grad():
                src_in_trg = [FDA_source_to_target_unet(x["image"], y["image"], self.unet_model, start_iter) for x, y in zip(label_data, unlabel_data)]
                label_data_aug = []
                for x,y in zip(label_data, src_in_trg):
                    z = x.copy()
                    #z['image'] = (x['image'].cuda() + (y['src_org'].cuda())*255)/2
                    #label_data_aug.append(z)
                    #print(x['image'].shape)
                    #print(y['image'].shape)
                    rgb_img_tensor = torch.mean(torch.stack((((x["image"]/255)*0.5).unsqueeze(0).cuda(), (y["src_org"]).unsqueeze(0))), dim=0)
                    #print(rgb_img_tensor.shape)
                    z['image'] = rgb_img_tensor.squeeze()
                    #print(rgb_img_tensor.shape)
                    #print(z['image'].shape)
                    label_data_aug.append(z)
                    rgb_img_tensor = rgb_img_tensor[:, [2, 1, 0], :, :]
                    #save_image(rgb_img_tensor,'./demo_images/'+"test_"+str(start_iter)+'.png')
                #label_data.extend(label_data_aug)
            

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            record_dict, _, _, _ = self.model(
                label_data, branch="supervised")
            
            if start_iter >5000:
                record_dict_night, _, _, _ = self.model(
                    label_data_aug, branch="supervised")
                
                temp_dict = {}
                for key in record_dict_night.keys():
                    temp_dict[key+'_night'] = record_dict_night[key]
                record_dict.update(temp_dict)
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        # Student-teacher stage
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            # 1. Input labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            #  2. Remove unlabeled data labels
            gt_unlabel = self.get_label(unlabel_data)
            unlabel_data = self.remove_label(unlabel_data)

            #  3. Generate the easy pseudo-label using teacher model (Phase-1)
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup,
                    proposals_roih_unsup,
                    _,
                ) = self.model_teacher(unlabel_data, branch="unsup_data_weak")

            #  4. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup
            #Process pseudo labels and thresholding
            (
                pesudo_proposals_rpn_unsup,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup, _ = self.process_pseudo_label(
                proposals_roih_unsup, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup

            # 5. Add pseudo-label to unlabeled data
            unlabel_data = self.add_label(
                unlabel_data, joint_proposal_dict["proposals_pseudo_roih"]
            )

            #6. Scale student inputs (pseudo-labels and image)
            if self.cfg.STUDENT_SCALE:
                scale_mask=np.where(self.iter<self.scale_checkpoints)[0]
                if len(scale_mask)>0:
                    nstu_scale = self.scale_list[scale_mask[0]]
                else:
                    nstu_scale = 1.0
                self.stu_scale = np.random.normal(nstu_scale,0.15)
                if self.stu_scale < 0.4:
                    self.stu_scale = 0.4
                elif self.stu_scale > 1.0:
                    self.stu_scale = 1.0
                scaled_unlabel_data = [x.copy() for x in unlabel_data]
                img_s = scaled_unlabel_data[0]['image'].shape[1:]
                self.scale_t = T.Resize((int(img_s[0]*self.stu_scale), int(img_s[1]*self.stu_scale)))
                for item in scaled_unlabel_data:
                    item['image'] = item['image'].cuda()
                    item['image']=self.scale_t(item['image'])
                    item['instances'].gt_boxes.scale(self.stu_scale,self.stu_scale)
                    if nstu_scale < 1.0:
                        gt_mask = item['instances'].gt_boxes.area()>16 #16*16
                    else:
                        gt_mask = item['instances'].gt_boxes.area()>16 #8*8
                    gt_boxes = item['instances'].gt_boxes[gt_mask]
                    gt_classes = item['instances'].gt_classes[gt_mask]
                    scores = item['instances'].scores[gt_mask]
                    item['instances'] = Instances(item['image'].shape[1:],gt_boxes=gt_boxes, gt_classes = gt_classes, scores=scores)
                
            else:
                # if student scaling is not used
                scaled_unlabel_data = [x.copy() for x in unlabel_data] 

            #7. Input scaled inputs into student
            (pseudo_losses, 
            proposals_into_roih, 
            rpn_stu,
            roi_stu,
            pred_idx)= self.model(
                scaled_unlabel_data, branch="consistency_target"
            )
            new_pseudo_losses = {}
            for key in pseudo_losses.keys():
                new_pseudo_losses[key + "_pseudo"] = pseudo_losses[
                    key
                ]
            record_dict.update(new_pseudo_losses)

            #8. Upscale student RPN proposals for teacher
            if self.cfg.STUDENT_SCALE:
                    stu_resized_proposals = []
                    for k,proposals in enumerate(proposals_into_roih):
                        stu_resized_proposals.append(Instances(scaled_unlabel_data[0]['image'].shape[1:],
                                                proposal_boxes = proposals.proposal_boxes.clone(),
                                                objectness_logits = proposals.objectness_logits,
                                                gt_classes = proposals.gt_classes,
                                                gt_boxes = proposals.gt_boxes))
                        stu_resized_proposals[k].proposal_boxes.scale(1/self.stu_scale,1/self.stu_scale)
                    proposals_into_roih=stu_resized_proposals
            
            #9. Generate matched pseudo-labels from teacher (Phase-2)
            with torch.no_grad():
                (_,
                _,
                roi_teach,
                _
                )= self.model_teacher(
                    unlabel_data, 
                    branch="unsup_data_consistency", 
                    given_proposals=proposals_into_roih, 
                    proposal_index=pred_idx
                )
                
            # print("roi stu: "+ str(roi_stu))
            # print("roi teach: "+ str(roi_teach))
            
            # 10. Compute consistency loss
            cons_loss = self.consistency_losses.losses(roi_stu,roi_teach)
            record_dict.update(cons_loss)

        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo": 
                    loss_dict[key] = record_dict[key] * 0
                elif key.endswith('loss_cls_pseudo'):
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                elif key.endswith('loss_rpn_cls_pseudo'):
                    loss_dict[key] = record_dict[key] 
                else: 
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        if self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP and self.cfg.STUDENT_SCALE:
            metrics_dict["scale"] = self.stu_scale
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()



    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

