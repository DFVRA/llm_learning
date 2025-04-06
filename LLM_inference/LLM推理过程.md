### LLM 的推理过程
LLM 的推理过程分为两个阶段：prefill 阶段 和 decode阶段
* prefill 阶段：模型对所有的 prompt token 一次并行计算，最终会产生第一个输出 token
* decode 阶段：每次生成一个 token, 直到生成 EOS(end-of-sequence) token，产生最终的 response
 
 在进行 token 生成的时候，t 位置的 q 只和 t 位置之前的 k v 计算，并且 k v 的值不受 t 后面 token 的影响，所以可以把计算好的 k v 缓存起来，将
 计算密集型任务转变为访存密集型任务
 
 现在的问题是 LLM size 比较大，GPU 的显存空间比较宝贵，通过显存来保存 k-v cache 会带来访存限制
 #### 模型推理阶段显存占用情况
 推理阶段主要有三部分数据会存放在显存里面：
 * KV cache:存储量会随着 Batch, Sequence_len 长度动态变化
 * 模型参数: 72 * 2 = 144GB
 * 运行时中间数据：推理过程中产生的中间数据会临时存储在显存，即用即释放，一般占用空间比较小
 
 以一个 token 的计算过程为例，看一下 一个 token 计算需要存储多少 K V. 以 Qwen-72B-chat 为例：模型共 80 层，每层有 64 个 head，每个 head 的维度为128
 计算一个 token, 每个 transformer 层的每个 head 都要存储一对 k v, 故需要存储 2 * 80 * 64 = 10240 个K V  如果推理阶段采用的是半精度（bf16）参数，
 每个参数占用 2B ，最终一个 token 的存储占用将达到 2 * 10240 * 128 = 2.62MB
    * 单条短文本场景: batch_size = 1 seq_len = 2048    mem = 2.62MB * 1 * 2048 = 5.366GB, 模型参数量占大头，使用 80G A100 至少需要两张卡做推理
    * 并发长文本场景: batch_size = 32 seq_len = 4096   mem = 2.62MB * 32 * 4096 = 343.4GB，使用 80G A100 至少需要七张卡做推理
    
 推理阶段根据离线，在线的业务场景，组多大的 batch，是一个 balance 的过程，batch 比较小，虽然并发度不高，但是可能单卡就可以装下完整的模型和KVCache，这时候卡内带宽比较高，
 性能比较出众，可以适当考虑增大 Batch把显卡内存打满，但是如果单卡无法部署，跨卡跨机访存性能会下降，GPU的计算资源使用效率不高。
 #### 共享 KV 优化显存
 ##### MQA(Multi-Query Attention)
 每一层的所有 hea共享一个 KV 来计算 attention，相对于 MHA 的单个 token 需要保存的 KV 数（layer_nums * num_heads * 2）,变成了（layer_nums * 2）
 ##### GQA(Group-Query Attention)
 GQA 是 MHA 和 MQA 的折中，对所有的 head 分组，如果分为 G组，那么 num_heads / G 个头共享一个 KV ，需要保存的 KV 数为：layer_nums * G * 2
 