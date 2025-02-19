import kagglehub
import random
import torch
from utils import create_data_lists, make_dataloader
from cnn import MushroomClassifier
from trainer import MushroomTrainer


def main():
    # important to keep seed same if you're loading datasets
    SEED = 81
    random.seed(SEED)
    torch.manual_seed(SEED)
    # BASE_DIR = kagglehub.dataset_download("maysee/mushrooms-classification-common-genuss-images")
    # print("Path to dataset files:", path)
    base_dir = "/Users/keerthireddy/.cache/kagglehub/datasets/maysee/mushrooms-classification-common-genuss-images/versions/1/Mushrooms"

    print("creating data lists")
    train_data, val_data, test_data, labels = create_data_lists(base_dir)
    num_classes = len({data["genus"] for data in train_data})

    print("creating dataloaders")
    # make dataloaders
    val_dataloader = make_dataloader(val_data)
    test_dataloader = make_dataloader(test_data)

    print("making neural net and trainer")

    shroom_classifier = MushroomClassifier(num_classes)
    shroom_trainer = MushroomTrainer(
        shroom_classifier,
        train_data,
        val_dataloader,
        test_dataloader,
        num_epochs=15,
        seed=SEED,
        labels=labels,
    )

    # if you want to load a checkpoint, do so here:
    # shroom_trainer.load_checkpoint("./models/model_0_epoch_0.pth", seed=SEED)
    print("training model")
    shroom_trainer.train_model()


if __name__ == "__main__":
    main()
