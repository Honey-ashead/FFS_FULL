import inspect
import logging
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from .roi_heads import ROIHeads, select_foreground_proposals, select_proposals_with_visible_keypoints
from FrEIA.framework import OutputNode, Node, InputNode, GraphINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
import numpy as np


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ROI_HEADS_REGISTRY.register()
class ROIHeadsLogisticGMM(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            keypoint_in_features: Optional[List[str]] = None,
            keypoint_pooler: Optional[ROIPooler] = None,
            keypoint_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.sample_number = self.cfg.FFS.SAMPLE_NUMBER
        self.start_iter = self.cfg.FFS.STARTING_ITER
        # print(self.sample_number, self.start_iter)

        self.logistic_regression = torch.nn.Linear(1, 2)
        self.logistic_regression.cuda()
        # torch.nn.init.xavier_normal_(self.logistic_regression.weight)

        self.select = 10
        self.sample_from = 10000
        self.loss_weight = 0.1
        self.weight_energy = torch.nn.Linear(self.num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.data_dict = torch.zeros(self.num_classes, self.sample_number, 1024).cuda()
        self.number_dict = {}
        self.eye_matrix = torch.eye(1024, device='cuda')
        self.trajectory = torch.zeros((self.num_classes, 900, 3)).cuda()
        for i in range(self.num_classes):
            self.number_dict[i] = 0
        self.cos = torch.nn.MSELoss()
        self.max_iterations = 24000

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        cls.cfg = cfg
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features, "mask_pooler": (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )}
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features, "keypoint_pooler": (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )}
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            iteration: int,
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, iteration)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, iteration)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], iteration: int):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training:
            scores, proposal_deltas = predictions
            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )
            _log_classification_stats(scores, gt_classes)
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                gt_boxes = cat(
                    [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                    dim=0,
                )

                sum_temp = 0
                for index in range(self.num_classes):
                    sum_temp += self.number_dict[index]
                lr_reg_loss = torch.zeros(1).cuda()

                if sum_temp == self.num_classes * self.sample_number and iteration < self.start_iter:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    # maintaining an ID data queue for each class.
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              box_features[index].detach().view(1, -1)), 0)
                elif sum_temp == self.num_classes * self.sample_number and iteration >= self.start_iter:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    # maintaining an ID data queue for each class.
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              box_features[index].detach().view(1, -1)), 0)
                    # the covariance finder needs the data to be centered.
                    for index in range(self.num_classes):
                        if index == 0:
                            X = self.data_dict[index] - self.data_dict[index].mean(0)
                            mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                        else:
                            X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                            mean_embed_id = torch.cat((mean_embed_id,
                                                       self.data_dict[index].mean(0).view(1, -1)), 0)

                    # add the variance.
                    temp_precision = torch.mm(X.t(), X) / len(X)
                    # for stable training.
                    temp_precision += 0.0001 * self.eye_matrix

                    for index in range(self.num_classes):
                        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                            mean_embed_id[index], covariance_matrix=temp_precision)
                        negative_samples = new_dis.rsample((self.sample_from,))
                        prob_density = new_dis.log_prob(negative_samples)

                        # keep the data in the low-density area.
                        cur_samples, index_prob = torch.topk(-prob_density, self.select)
                        if index == 0:
                            ood_samples = negative_samples[index_prob]
                        else:
                            ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        del new_dis
                        del negative_samples

                    if len(ood_samples) != 0:
                        # add some gaussian noise
                        # ood_samples = self.noise(ood_samples)
                        energy_score_for_fg = self.log_sum_exp(predictions[0][selected_fg_samples][:, :-1], 1)
                        predictions_ood = self.box_predictor(ood_samples)
                        # # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                        energy_score_for_bg = self.log_sum_exp(predictions_ood[0][:, :-1], 1)

                        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                        labels_for_lr = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                                   torch.zeros(len(ood_samples)).cuda()), -1)
                        weights_fg_bg = torch.Tensor([len(selected_fg_samples) / float(len(input_for_lr)),
                                                      len(ood_samples) / float(len(input_for_lr))]).cuda()
                        criterion = torch.nn.CrossEntropyLoss()  # weight=weights_fg_bg)
                        output = self.logistic_regression(input_for_lr.view(-1, 1))
                        lr_reg_loss = criterion(output, labels_for_lr.long())

                    del ood_samples

                else:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        if self.number_dict[dict_key] < self.sample_number:
                            self.data_dict[dict_key][self.number_dict[dict_key]] = box_features[index].detach()
                            self.number_dict[dict_key] += 1
                # create a dummy in order to have all weights to get involved in for a loss.
                loss_dummy = self.cos(self.logistic_regression(torch.zeros(1).cuda()), self.logistic_regression.bias)
                loss_dummy1 = self.cos(self.weight_energy(torch.zeros(self.num_classes).cuda()),
                                       self.weight_energy.bias)
                del box_features
                # print(self.number_dict)

            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

            if sum_temp == self.num_classes * self.sample_number:
                losses = {
                    "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                    "lr_reg_loss": self.loss_weight * lr_reg_loss,
                    "loss_dummy": loss_dummy,
                    "loss_dummy1": loss_dummy1,
                    "loss_box_reg": self.box_predictor.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                    ),
                }
            else:
                losses = {
                    "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                    "lr_reg_loss": torch.zeros(1).cuda(),
                    "loss_dummy": loss_dummy,
                    "loss_dummy1": loss_dummy1,
                    "loss_box_reg": self.box_predictor.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                    ),
                }
            losses = {k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()}

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            # https://github.com/pytorch/pytorch/issues/49728
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            # https://github.com/pytorch/pytorch/issues/49728
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = dict([(f, features[f]) for f in self.keypoint_in_features])
        return self.keypoint_head(features, instances)


@ROI_HEADS_REGISTRY.register()
class ROIHeadsFFSLogisticGMM(ROIHeadsLogisticGMM):
    @configurable
    def __init__(self, *,
                 box_in_features: List[str],
                 box_pooler: ROIPooler,
                 box_head: nn.Module,
                 box_predictor: nn.Module,
                 **kwargs):
        super(ROIHeadsFFSLogisticGMM, self).__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            **kwargs)
        # Stack all the coupling blocks including the permute blocks
        self.in1 = InputNode(1024, name='input1')
        self.layer1 = Node(self.in1, GLOWCouplingBlock, {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                           name=F'coupling_{0}')
        self.layer2 = Node(self.layer1, PermuteRandom, {'seed': 0}, name=F'permute_{0}')
        self.layer3 = Node(self.layer2, GLOWCouplingBlock, {'subnet_constructor': self.subnet_fc, 'clamp': 2.0},
                           name=F'coupling_{1}')
        self.layer4 = Node(self.layer3, PermuteRandom, {'seed': 1}, name=F'permute_{1}')
        self.out1 = OutputNode(self.layer4, name='output1')
        self.flow_model = GraphINN([self.in1, self.layer1, self.layer2, self.layer3, self.layer4, self.out1])

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        cls.cfg = cfg
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(
            nn.Linear(c_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, c_out)
        )

    def NLLLoss(self, z, sldj, mean=True):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) - np.log(256) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll
        if mean:
            return nll.mean()
        return nll

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], iteration: int):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # self.update_feature_store(box_features, proposals)
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        box_predictor_loss = 0.
        if self.training:
            # self.update_feature_store(box_features, predictions)
            scores, proposal_deltas = predictions
            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )

            _log_classification_stats(scores, gt_classes)
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"

                sum_temp = 0
                for index in range(self.num_classes):
                    sum_temp += self.number_dict[index]
                lr_reg_loss = torch.zeros(1).cuda()

                selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                z, sldj = self.flow_model(box_features[indices_numpy,].detach().cuda())
                nll_loss = self.NLLLoss(z, sldj)

                self.box_predictor.update_feature_store(box_features[indices_numpy], gt_classes[indices_numpy])
                box_predictor_loss = self.box_predictor.losses(predictions, proposals, box_features[indices_numpy])
                if iteration >= self.start_iter:
                    # randomly sample from latent space of the flow model
                    with torch.no_grad():
                        z_randn = torch.randn((self.sample_from, 1024), dtype=torch.float32).cuda()
                        negative_samples, _ = self.flow_model(z_randn, rev=True)
                        _, sldj_neg = self.flow_model(negative_samples)
                        nll_neg = self.NLLLoss(z_randn, sldj_neg, mean=False)
                        cur_samples, index_prob = torch.topk(nll_neg, self.select)
                        ood_samples = negative_samples[index_prob].view(self.select, -1)
                        ood_indices = self.box_predictor.get_ood_indices(ood_samples)
                        if ood_indices.size(0) == 0:
                            cur_samples, index_prob = torch.topk(nll_neg, 1)
                            ood_samples = negative_samples[index_prob].view(1, -1)
                        else:
                            ood_samples = ood_samples[ood_indices]

                        del negative_samples
                        del z_randn

                    if len(ood_samples) != 0:
                        energy_score_for_fg = self.log_sum_exp(predictions[0][selected_fg_samples][:, :-1], 1)
                        predictions_ood = self.box_predictor(ood_samples)
                        energy_score_for_bg = self.log_sum_exp(predictions_ood[0][:, :-1], 1)

                        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                        labels_for_lr = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                                   torch.zeros(len(ood_samples)).cuda()), -1)
                        # weights_fg_bg = torch.Tensor([len(selected_fg_samples) / float(len(input_for_lr)),
                        # len(ood_samples) / float(len(input_for_lr))]).cuda()
                        criterion = torch.nn.CrossEntropyLoss()  # weight=weights_fg_bg)
                        output = self.logistic_regression(input_for_lr.view(-1, 1))
                        lr_reg_loss = criterion(output, labels_for_lr.long())

                    del ood_samples

                # create a dummy in order to have all weights to get involved in for a loss.
                loss_dummy = self.cos(self.logistic_regression(torch.zeros(1).cuda()), self.logistic_regression.bias)
                loss_dummy1 = self.cos(self.weight_energy(torch.zeros(self.num_classes).cuda()),
                                       self.weight_energy.bias)
                del box_features

            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "lr_reg_loss": self.loss_weight * lr_reg_loss,
                "loss_nll": 1e-4 * nll_loss,
                "loss_dummy": loss_dummy,
                "loss_dummy1": loss_dummy1,
            }
            if self.training:
                losses.update(box_predictor_loss)

            losses = {k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()}

            # proposals are modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            iteration: int,
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # self.update_feature_store(box_features, predictions)
            losses = self._forward_box(features, proposals, iteration)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, iteration)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
