# SFT 相关知识点

## 问题一：LoRA 嵌入在什么结构中？数学原理是什么？

**回答：**

本项目设置 `target_modules=all-linear`，LoRA 会注入 Qwen 的线性层，主要包括 Attention 的 `q_proj`、`k_proj`、`v_proj`、`o_proj` 和 MLP 的 `gate_proj`、`up_proj`、`down_proj`。

原始线性层为：

$$
y=Wx
$$

加入 LoRA 后：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

其中原始权重 $W$ 冻结，只训练低秩矩阵 $A$ 和 $B$。因为秩 $r$ 很小，所以训练参数量和显存占用大幅减少。

$r$ 是 LoRA 的秩，决定低秩空间的维度。$r$ 越大，LoRA 的表达能力和可训练参数量越大，同时显存占用也越高。本项目设置 $r=64$。

$\alpha$ 是 LoRA 的缩放系数，控制 LoRA 更新对原模型输出的影响强度。PEFT 默认使用 $\alpha/r$ 进行缩放，本项目设置 $\alpha=32$，因此缩放系数为：

$$
\frac{\alpha}{r}=\frac{32}{64}=0.5
$$

## 问题二：什么是交叉熵损失？为什么 SFT 使用它？还有哪些损失函数？

**回答：**

语言模型在每个位置都要从整个词表中预测下一个 token，本质上是多分类问题。若正确 token 的预测概率为 $p_y$，交叉熵为：

$$
L=-\log p_y
$$

正确 token 的概率越高，loss 越小。交叉熵直接对应下一个 token 的最大似然训练，因此适合 SFT。本项目还通过 `loss_mask` 只保留回答部分的 loss。

其他常见损失函数包括：

- MSE、MAE：回归任务
- BCE：二分类或多标签分类
- KL 散度：知识蒸馏、模型分布约束
- Contrastive Loss：表征学习
- Ranking Loss：偏好排序、奖励模型

## 问题三：为什么反向传播后还需要优化器？冻结参数会怎样？

**回答：**

`loss.backward()` 只负责计算梯度，`optimizer.step()` 才负责根据梯度更新参数。

直接执行：

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

也是可以的，这就是最基本的 SGD。AdamW 在此基础上加入动量、自适应学习率和权重衰减，通常更适合大模型训练。

冻结的 Qwen 参数仍参与前向计算，梯度也能经过这些运算继续向前传播，但不会保存和更新这些参数的梯度。只有 `requires_grad=True` 的 LoRA 参数会获得梯度，并被优化器更新。

## 问题四：`all-linear` 的实际覆盖范围是什么？

**回答：**

`target_modules=all-linear` 表示给每个 Transformer Decoder Layer 中符合条件的线性层都加入独立的 LoRA 旁路，而不是只修改少数几层。

Qwen2.5-7B-Instruct 有 28 个 Decoder Layer，每层覆盖：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- MLP：`gate_proj`、`up_proj`、`down_proj`

因此每层有 7 个 LoRA 模块，共有：

$$
28\times7=196
$$

Embedding、RMSNorm、RoPE、Softmax、SiLU、残差连接和最终 `lm_head` 不在覆盖范围内。
