import config
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from train import SpellCheckTrainer, IntentClassifyTrainer
from preprocess import SpellCheckProcessor, IntentClassifyProcessor
from models_def import SpellCheckModel, IntentClassifyModel, load_params

learning_rate = 1e-5
device = config.DEVICE
bert = config.PRETRAINED_DIR / "bert-base-chinese"
batch_size = 64

def model_go(task, train=0, test=0, inference=0, model_params_path=None):
    match task:
        case "spell_check":
            model = SpellCheckModel(bert)
            processor = SpellCheckProcessor(
                data_path=config.SPELL_CHECK_RAW_DATA_DIR / "data.txt",
                save_dir=config.SPELL_CHECK_PROCESSED_DATA_DIR,
                tokenizer=model.tokenizer,
                max_seq_len=512,
            )
            trainer = SpellCheckTrainer(model, device, 2, learning_rate)
        case "intent_classify":
            model = IntentClassifyModel(bert, config.INTENT_INFO)
            processor = IntentClassifyProcessor(
                data_path=config.INTENT_CLASSIFY_RAW_DATA_DIR / "data.jsonl",
                save_dir=config.INTENT_CLASSIFY_PROCESSED_DATA_DIR,
                tokenizer=model.tokenizer,
                max_seq_len=512,
                intent_info=config.INTENT_INFO,
            )
            trainer = IntentClassifyTrainer(model, device, 10, learning_rate)
        case _:
            raise ValueError("任务须为 spell_check 或 intent_classify")

    save_name = f"{task}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    load_params(model, model_params_path)
    writer = None

    if train:
        writer = SummaryWriter(config.LOGS_DIR / save_name)
        dataloader = {
            "train": processor.get_dataloader("train", batch_size),
            "valid": processor.get_dataloader("valid", batch_size),
        }
        model_params_path = config.FINETUNED_DIR / f"{save_name}.pt"
        trainer(dataloader, model_params_path, writer)

    if test:
        test_dataloader = processor.get_dataloader("test", batch_size)
        trainer({"test": test_dataloader}, writer=writer, is_test=True)

    if writer:
        writer.close()

    if inference:
        print(model.predict(text, device))

text = [
    "高血压常见的治疗方法有哪些？",
    "甲状腺结节环者日常饮食有什么推荐吗？",
    "我想预约明天张主任的专家号",
]

model_go("spell_check", 1, 1, 1, config.FINETUNED_DIR / "spell_check.pt")
model_go("intent_classify", 1, 1, 1, config.FINETUNED_DIR / "intent_classify.pt")
