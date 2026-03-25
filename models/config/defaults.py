from detectron2.config import CfgNode as CN


# 向Detectron2框架的配置系统添加模型所需的自定义配置项
def add_opendet_config(cfg):
    _C = cfg

    # unknown probability loss 未知概率损失
    # 用于处理开放集检测中的未知类别样本，通过特定的损失函数让模型学会区分已知和未知类别。
    _C.UPLOSS = CN()
    _C.UPLOSS.START_ITER = 100  # 损失开始的迭代次数（通常与 warmup 迭代次数相同）
    _C.UPLOSS.SAMPLING_METRIC = "min_score" # 采样指标
    _C.UPLOSS.TOPK = 3  # 选择 top-k 个样本
    _C.UPLOSS.ALPHA = 1.0 # 平衡参数
    _C.UPLOSS.WEIGHT = 0.5 # 损失权重

    # instance contrastive loss 实例对比损失
    #通过对比学习增强特征表示能力，使用记忆队列存储正负样本特征。
    _C.ICLOSS = CN()
    _C.ICLOSS.OUT_DIM = 128  # 输出特征维度
    _C.ICLOSS.QUEUE_SIZE = 256 # 特征队列大小
    _C.ICLOSS.IN_QUEUE_SIZE = 16 # 实例队列大小
    _C.ICLOSS.BATCH_IOU_THRESH = 0.5 # batch 内的 IoU 阈值
    _C.ICLOSS.QUEUE_IOU_THRESH = 0.7 # 队列内的 IoU 阈值
    _C.ICLOSS.TEMPERATURE = 0.1 # 温度系数（用于对比学习）
    _C.ICLOSS.WEIGHT = 0.1 # 损失权重

    # register RoI output layer ROI Head 相关配置
    _C.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS = "FastRCNNOutputLayers" # ROI 输出层类型
    # known classes
    _C.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 20    # 已知类别数量
    _C.MODEL.RETINANET.NUM_KNOWN_CLASSES = 20 # RetinaNet 的已知类别数
    # thresh for visualization results.
    _C.MODEL.ROI_HEADS.VIS_IOU_THRESH = 1.0  # 可视化结果的 IoU 阈值
    # scale for cosine classifier
    _C.MODEL.ROI_HEADS.COSINE_SCALE = 20     # 余弦分类器的缩放因子

    # swin transformer  Swin Transformer 骨干网络配置
    _C.MODEL.SWINT = CN()
    _C.MODEL.SWINT.EMBED_DIM = 96    # 嵌入维度
    _C.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"] # 输出特征层
    _C.MODEL.SWINT.DEPTHS = [2, 2, 6, 2] # 各阶段层数
    _C.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24] # 各阶段注意力头数
    _C.MODEL.SWINT.WINDOW_SIZE = 7 # 窗口大小
    _C.MODEL.SWINT.MLP_RATIO = 4 # MLP 扩展比例
    _C.MODEL.SWINT.DROP_PATH_RATE = 0.2 # DropPath 比率
    _C.MODEL.SWINT.APE = False # 是否使用绝对位置编码
    _C.MODEL.BACKBONE.FREEZE_AT = -1 # 冻结骨干网络的层级（-1 表示不冻结）
    _C.MODEL.FPN.TOP_LEVELS = 2 # FPN 顶层级别

    # solver, e.g., adamw for swin  优化器配置
    _C.SOLVER.OPTIMIZER = 'SGD'  # 优化器类型
    _C.SOLVER.BETAS = (0.9, 0.999) # Adam/AdamW 的 beta 参数
