# 作业五：LMDeploy量化部署LLM & VLM实践

### 基础作业：LMDeploy环境部署

创建conda环境：

![alt text](imgs/环境配置.png)

安装LMDeploy：

![alt text](imgs/安装lmdeploy.png)

### 基础作业：LMDeploy模型对话（chat）

还是用软链接的方式声明存放模型的目录，随后使用Transformer库直接运行模型，生成两个response共计用时约19s

![alt text](imgs/transformer库推理.png)

然后使用lmdeploy进行推理，生成两个response共计用时约9s，明显更快

![alt text](imgs/lmdeploy推理.png)

