import os
import torch
import random
from datasets import load_from_disk, Dataset
from torch.utils.data import Subset, DataLoader

class Processor:
    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        max_seq_len,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def process(self):
        """处理数据集并保存"""
        dataset = self._make_dataset()
        dataset = self._split_dataset(dataset)
        # 保存数据集
        for type in ["train", "valid", "test"]:
            dataset[type].save_to_disk(self.save_dir / type)

    def get_dataloader(self, type, batch_size, max_examples=0):
        """获取保存的数据集，并加载 Dataloader"""
        if not os.path.exists(self.save_dir / type):
            self.process()
        # 加载数据集
        dataset = load_from_disk(self.save_dir / type)
        # 如果限制了最大样本数，获取不多于 max_examples 的样本
        if max_examples:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            max_examples = min(max_examples, len(dataset))
            indices = indices[:max_examples]
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, batch_size, shuffle=(type == "train"))

    def _make_dataset(self):
        """处理数据集"""
        raise NotImplementedError

    def _split_dataset(self, dataset):
        """划分数据集"""
        train_size = int(dataset.num_rows * self.train_ratio)
        dataset = dataset.train_test_split(test_size=self.test_ratio)
        train_valid_split = dataset["train"].train_test_split(train_size=train_size)
        dataset["train"] = train_valid_split["train"]
        dataset["valid"] = train_valid_split["test"]
        return dataset

class SpellCheckProcessor(Processor):
    """拼写检查任务数据预处理"""

    def _make_dataset(self):
        """处理数据集"""
        dataset = Dataset.from_generator(self._generate_examples)
        dataset = dataset.map(
            self._map_fn, batched=True, remove_columns=["text", "label"]
        )
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def _generate_examples(self):
        # 使用生成器，读取数据集，逐步生成结果
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    pair = line.split()
                    if len(pair) == 2:
                        yield {"text": pair[0], "label": pair[1]}

    def _map_fn(self, examples):
        # 处理数据集，将文本转换为模型输入
        inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        labels = self.tokenizer(
            examples["label"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class IntentClassifyProcessor(Processor):
    """意图分类任务数据预处理"""

    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        max_seq_len,
        intent_info,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        super().__init__(
            data_path, save_dir, tokenizer, max_seq_len, train_ratio, test_ratio
        )
        self.intent_level1_list = sorted(list(intent_info.keys()))
        intent_level2_lists = [
            sorted(intent_info[intent_level1])
            for intent_level1 in self.intent_level1_list
        ]
        self.intent_level1_map = {v: k for k, v in enumerate(self.intent_level1_list)}
        self.intent_level2_map_list = [
            {v: k for k, v in enumerate(intent_level2_list)}
            for intent_level2_list in intent_level2_lists
        ]

    def _make_dataset(self):
        """处理数据集"""
        dataset = Dataset.from_json(str(self.data_path))
        dataset = dataset.map(
            self._map_fn,
            batched=True,
            remove_columns=["text", "intent", "request", "consult"],
        )
        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "intent_level1_labels",
                "intent_level2_labels",
            ],
        )
        return dataset

    def _map_fn(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        intent_level1_labels = self._label_2_onehot(
            examples["intent"], self.intent_level1_map
        )
        intent_level2_label_lists = [
            self._label_2_onehot(examples[intent_level1], intent_level2_map)
            for intent_level1, intent_level2_map in zip(
                self.intent_level1_list, self.intent_level2_map_list
            )
        ]
        intent_level2_labels = torch.cat(intent_level2_label_lists, dim=1)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "intent_level1_labels": intent_level1_labels,
            "intent_level2_labels": intent_level2_labels,
        }

    def _label_2_onehot(self, batch_label_list: list[list[str]], label_map: dict):
        """将标签列表转换为 one-hot 标签"""
        batch_size = len(batch_label_list)
        num_classes = len(label_map)
        onehot = torch.zeros((batch_size, num_classes), dtype=torch.int)
        for i, label_list in enumerate(batch_label_list):
            for label in label_list:
                onehot[i][label_map[label]] = 1
        return onehot
