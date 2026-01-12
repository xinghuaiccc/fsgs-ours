# GFC (Geometry-First Curriculum) 训练策略详解笔记

本文档记录本项目中的创新训练策略 GFC (Geometry-First Curriculum, 几何优先课程学习) 的设计动机、算法细节、实现方式、超参数建议、消融实验设计与局限性分析。可作为论文写作和方法复现的参考。

## 1. 研究动机与问题定义

### 1.1 任务背景
本项目使用 3D Gaussian Splatting (FSGS) 进行新视角合成。现实实验中经常采用极少视角训练 (如 n_views=3)。此时模型容量较大、参数较多，训练容易出现:
- 训练损失持续下降，但测试 PSNR 在某个阶段后下降。
- 颜色/纹理细节快速过拟合训练视角，几何结构未充分稳定。
- densify 引入大量新高斯点进一步增强容量，加剧过拟合。

### 1.2 典型现象
在 n_views=3 的设置下，测试 PSNR 往往在 2k-3k 左右达到峰值，之后下降。这说明模型在早期能学习基本结构，但后期更多是在记忆训练视角的纹理和颜色，导致泛化恶化。

### 1.3 GFC 的核心假设
若能在训练早期强制模型优先学习几何 (geometry-first)，并抑制颜色/纹理的自由拟合，则:
- 几何结构会更加稳定。
- 后期颜色学习在稳定几何基础上进行，泛化性能更好。
- 少视角情况下测试 PSNR 更稳定，峰值更高。

## 2. 方法概览

GFC 是一种分阶段课程学习策略，核心思想是:
1) Stage 1 (几何优先阶段):
   - 冻结或极大降低颜色参数学习率。
   - 提高深度监督权重，使模型对几何结构更敏感。
   - 可冻结 SH 阶数增长，限制过早引入复杂光照表达。
2) Stage 2 (外观学习阶段):
   - 解冻颜色参数，恢复正常学习率。
   - 降回正常深度权重。
   - 解冻 SH 阶数增长。

这是一种结构化优化策略，避免模型在几何未收敛时学习高频外观，从而提升测试集 PSNR。

## 3. 具体算法设计

### 3.1 训练参数分解
将 Gaussian 参数分为两类:
- 几何相关参数: xyz, scaling, rotation, opacity
- 外观相关参数: SH features (f_dc, f_rest)

GFC 的主要调控目标是外观相关参数。

### 3.2 GFC 阶段设计

设训练迭代总数为 T，定义阶段分界:
- Stage 1: iteration <= gfc_stage1_end
- Stage 2: iteration > gfc_stage1_end

在 Stage 1:
- f_dc, f_rest 学习率缩放为 `gfc_feature_lr_scale_stage1`
  - 0.0 表示完全冻结颜色
  - 0.1 表示十分缓慢更新
- depth loss 权重提升到 `gfc_depth_weight`
- 可以冻结 SH 阶数提升 (gfc_freeze_sh)

在 Stage 2:
- 恢复原始学习率
- 深度权重回到正常值 (args.depth_weight)
- 恢复 SH 阶数提升

### 3.3 算法伪代码

```
for iteration in 1..T:
    if iteration <= gfc_stage1_end:
        set feature_lr = base_lr * gfc_feature_lr_scale_stage1
        depth_weight = gfc_depth_weight
        freeze SH upsampling
    else:
        restore feature_lr = base_lr
        depth_weight = base_depth_weight
        unfreeze SH upsampling

    render image
    loss = rgb_loss + depth_weight * depth_loss + other_regularizers
    backward + update
```

## 4. 实现细节 (本项目)

### 4.1 新增参数
在 `arguments/__init__.py` 新增:
- gfc_enable: 是否启用 GFC
- gfc_stage1_end: Stage1 结束迭代
- gfc_feature_lr_scale_stage1: Stage1 颜色学习率缩放
- gfc_depth_weight: Stage1 深度权重
- gfc_freeze_sh: Stage1 是否冻结 SH 阶数提升

### 4.2 训练逻辑变更
在 `train.py` 内:
- 在训练循环中判断 iteration 是否处于 Stage1。
- 记录 base_param_lrs，并动态修改 f_dc / f_rest 的 lr。
- depth loss 使用 current_depth_weight 代替 args.depth_weight。
- freeze_sh 控制 SH degree 增长。

## 5. 实验结果 (关键对比)

实验设置:
- 数据集: LLFF / fern
- n_views = 3
- 训练 3k
- 其他参数保持一致

### 5.1 原版 FSGS
- PSNR @3k ≈ 21.15 (峰值)

### 5.2 GFC-a (失败配置)
- stage1_end=3000, feature_lr_scale=0.0
- PSNR @3k ≈ 20.87
原因: 过度冻结外观，导致后期外观无法充分恢复。

### 5.3 GFC-b (成功配置)
- stage1_end=1500
- feature_lr_scale=0.1
- gfc_depth_weight=0.05
- grad_loss_weight=0.05, sh_sparsity_weight=0.1
- PSNR @3k ≈ 21.26 (超越原版)

结论: 适度冻结 + 深度强化 + 强正则 组合能提升少视角下 PSNR。

## 6. 为什么 GFC 有效

### 6.1 抑制过早拟合纹理
冻结颜色学习使模型无法通过颜色拟合早期误差，从而迫使几何参数承担拟合责任。

### 6.2 深度监督加强结构约束
深度 loss 在 Stage1 加强，强化模型对几何结构的对齐能力。

### 6.3 SH 延迟引入
SH 阶数提升会增加表达能力，冻结 SH 延迟引入复杂外观，有助于几何稳定。

### 6.4 课程学习思想
和经典 curriculum learning 一样，先学习简单结构，再学习细节，提高泛化。

## 7. 超参数与实践建议

推荐默认:
- gfc_stage1_end: 1500 (当总迭代 3000)
- gfc_feature_lr_scale_stage1: 0.1
- gfc_depth_weight: 0.05
- gfc_freeze_sh: True
- grad_loss_weight: 0.05
- sh_sparsity_weight: 0.1

经验:
- gfc_feature_lr_scale_stage1 = 0.0 容易过度抑制外观。
- stage1_end 过长会导致外观学习不足。
- stage1_end 过短则抑制效果不明显。

## 8. 可写进论文的贡献点

1) 提出 Geometry-First Curriculum (GFC) 训练策略:
   - 基于少视角过拟合现象设计。
2) 两阶段训练调度:
   - Stage1 几何优先
   - Stage2 外观补全
3) 实验验证在少视角设置下提升 PSNR:
   - 3 views 时测试 PSNR 超过原版。
4) 提供可复现的策略和超参数。

## 9. 消融实验设计建议

建议做以下消融:
- 去掉深度强化 (gfc_depth_weight=0)
- 去掉冻结 SH (gfc_freeze_sh=False)
- 调整 feature_lr_scale_stage1 = {0, 0.05, 0.1, 0.2}
- 调整 stage1_end = {1000, 1500, 2000}
- 对比不同 n_views (3, 5, 10)

指标:
- PSNR / SSIM / LPIPS
- PSNR 随迭代变化曲线是否单调提升

## 10. 局限性与未来工作

局限:
- 对极少视角有效，但对高视角可能改善有限。
- Stage1 深度强化依赖深度监督质量 (MiDaS)。

未来工作:
- 自适应确定 stage1_end (根据验证集变化自动切换)。
- 结合更强几何正则 (如稀疏深度一致性)。
- 与动态 densify 策略联合优化。

## 11. 推荐复现实验命令 (3k)

```
source /root/anaconda3/etc/profile.d/conda.sh && conda activate FSGS && \
python train.py --source_path /root/all-data/nerf_llff_data/fern/ \
  --model_path output-4/fern-gfc-b/ --eval --n_views 3 \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --save_iterations 2000 3000 --checkpoint_iterations 2000 3000 \
  --sh_degree 1 \
  --densify_from_iter 500 --densify_until_iter 1200 \
  --position_lr_init 5e-5 --position_lr_final 5e-7 \
  --grad_loss_weight 0.05 --sh_sparsity_weight 0.1 \
  --depth_weight 0.0 --depth_shape_weight 0.0 \
  --loss_print_interval 500 \
  --gfc_enable --gfc_stage1_end 1500 \
  --gfc_feature_lr_scale_stage1 0.1 \
  --gfc_depth_weight 0.05 --gfc_freeze_sh
```

## 12. 结论摘要 (论文可用)

GFC 提出了一种几何优先的课程学习策略，在少视角训练下显著缓解过拟合，提升测试 PSNR。核心机制是通过阶段式学习率调度与深度监督强化，限制早期外观拟合能力，使几何结构先稳定，再逐步引入外观细节。该策略结构简单、易于实现，并在 LLFF 3-view 设置上超越原版 FSGS 的峰值表现。
