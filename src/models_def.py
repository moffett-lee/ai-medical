import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

def load_params(model, model_params_path):
    """加载模型权重"""
    try:
        model.load_state_dict(torch.load(model_params_path))
    except (FileNotFoundError, AttributeError):
        print("指定的模型权重文件不存在，使用默认权重")
    except RuntimeError:
        model.load_state_dict(torch.load(model_params_path, map_location="cpu"))

class SpellCheckModel(nn.Module):
    """拼写检查模型"""

    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # 初始化 lm_head 并共享嵌入层权重
        hidden_size = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask)
        logits = self.lm_head(output.last_hidden_state)

        loss = 0.0
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        text: str | list[str],
        device=torch.device("cpu"),
        max_seq_len=512,
        batch_size=8,
    ):
        self.eval()
        self.to(device)

        res: list[str] = []
        # 统一转换为列表
        input_texts = text if isinstance(text, list) else [text]
        # 逐批次处理
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                max_length=max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = self.forward(inputs["input_ids"], inputs["attention_mask"])
            pred_ids = torch.argmax(outputs["logits"], dim=-1)
            # 将 attention_mask 为 0 的位置替换为 pad_token_id
            pred_ids[inputs["attention_mask"] == 0] = self.tokenizer.pad_token_id
            # 批量解码
            batch_res = self.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            # 去除空格
            batch_res = [i.replace(" ", "") for i in batch_res]
            res.extend(batch_res)

        # 输入字符串则返回字符串，输入列表则返回列表
        return res if isinstance(text, list) else res[0]

class IntentClassifyModel(nn.Module):
    """意图识别模型"""

    def __init__(self, model_name, intent_info: dict):
        super().__init__()
        self.intent_info = intent_info
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        hidden_dim = self.model.config.hidden_size
        # 一级意图分类层
        intent_1_linear_dim = len(intent_info)
        self.intent_1_linear = nn.Linear(hidden_dim, intent_1_linear_dim)
        # 二级意图分类层
        intent_2_linear_dim = sum([len(i) for i in intent_info.values()])
        self.intent_2_linear = nn.Linear(hidden_dim, intent_2_linear_dim)

        # 损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        intent_level1_labels=None,
        intent_level2_labels=None,
    ):
        """
        前向传播

        参数:
        - input_ids: 输入
        - attention_mask: 注意力掩码
        - intent_level1_labels: 一级意图标签
        - intent_level2_labels: 二级意图标签
        """
        outputs = self.model(input_ids, attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        intent_level1_logits = self.intent_1_linear(cls_hidden)
        intent_level2_logits = self.intent_2_linear(cls_hidden)

        loss = 0.0

        # 计算一级类别的损失
        if intent_level1_labels is not None:
            loss += self.loss_fn(intent_level1_logits, intent_level1_labels.float())

        # 计算二级类别的损失
        if intent_level2_labels is not None:
            loss += self.loss_fn(intent_level2_logits, intent_level2_labels.float())

        return {
            "intent_level1_logits": intent_level1_logits,
            "intent_level2_logits": intent_level2_logits,
            "loss": loss,
        }

    @torch.inference_mode()
    def predict(self, text, device=torch.device("cpu"), max_seq_len=512, batch_size=8):
        self.eval()
        self.to(device)

        # 结果列表
        intent_level1_list, intent_level2_lists = self.gen_label_lists(self.intent_info)
        num_res = len(text) if type(text) is list else 1
        res = [{} for _ in range(num_res)]

        bounds = self.gen_label_bound(intent_level2_lists)
        transfer_matrix = self.gen_mask_transfer_matrix(bounds).to(device)

        # 输入统一转换为列表
        input_texts = text if isinstance(text, list) else [text]

        # 逐批次处理
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                max_length=max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = self(inputs["input_ids"], inputs["attention_mask"])

            intent_level1_logits = outputs["intent_level1_logits"]
            intent_level1_preds = (torch.sigmoid(intent_level1_logits) > 0.5).float()

            label_mask = intent_level1_preds @ transfer_matrix
            intent_level2_logits = outputs["intent_level2_logits"] * label_mask
            intent_level2_preds = torch.sigmoid(intent_level2_logits) > 0.5

            for i, intent_level2_pred in enumerate(intent_level2_preds):
                for intent_level1, intent_level2_list, bound in zip(
                    intent_level1_list, intent_level2_lists, bounds
                ):
                    this_intent_level2_ids = (
                        intent_level2_pred[bound[0] : bound[1]]
                        .nonzero(as_tuple=True)[0]
                        .tolist()
                    )
                    if this_intent_level2_ids:
                        res[i][intent_level1] = [
                            intent_level2_list[idx] for idx in this_intent_level2_ids
                        ]
        return res if isinstance(text, list) else res[0]

    def gen_label_lists(self, intent_info):
        """按序生成每个级别的标签列表"""
        intent_level1_list = sorted(list(intent_info.keys()))
        intent_level2_lists = [
            sorted(self.intent_info[intent_level1])
            for intent_level1 in intent_level1_list
        ]
        return intent_level1_list, intent_level2_lists

    def gen_label_bound(self, intent_lists: list[list]):
        """生成每个次级标签在该层级中的边界"""
        bounds = [(0, len(intent_lists[0]))]
        for i in range(1, len(intent_lists)):
            bounds.append((bounds[-1][1], bounds[-1][1] + len(intent_lists[i])))
        return bounds

    def gen_mask_transfer_matrix(self, bounds):
        """生成用于生成掩码的矩阵"""
        transfer_matrix = torch.zeros((len(bounds), bounds[-1][1]), dtype=torch.float)
        for i, bound in enumerate(bounds):
            transfer_matrix[i][bound[0] : bound[1]] = 1
        return transfer_matrix
