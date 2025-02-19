import torch
import os
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time
import gc
from utils import make_dataloader


class MushroomTrainer:
    def __init__(
        self,
        model,
        train_data,
        val_dataloader,
        test_dataloader,
        num_epochs,
        seed,
        labels,
    ):
        self.seed = seed
        self.model = model
        self.train_data = train_data
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader = None
        self.labels = labels

        self.num_epochs = num_epochs

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        # self.optimizer = Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        # self.learning_rate_scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",  # Reduce LR when monitored value stops decreasing
        #     factor=0.7,
        #     patience=3,
        #     verbose=True,
        #     min_lr=1e-6,
        # )
        self.learning_rate_scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.save_dir = None

    def train_model(self):
        start_time = time.perf_counter()
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_val_accuracies = []
        epoch_val_top3_accuracies = []
        for epoch in range(self.num_epochs):
            # make a new training dataloader with diff augmentations each epoch
            self.train_dataloader = make_dataloader(
                self.train_data,
                num_augmentations=2,
                num_workers=2,
            )
            epoch_start = time.perf_counter()

            # train
            train_start = time.perf_counter()
            train_loss = self.train_epoch()
            train_time = time.perf_counter() - train_start
            epoch_train_losses.append(train_loss)

            # validate
            val_start = time.perf_counter()
            val_loss, val_accuracy, val_top3_accuracy = self.run_model(
                self.val_dataloader
            )
            val_time = time.perf_counter() - val_start
            epoch_val_losses.append(val_loss)
            epoch_val_accuracies.append(val_accuracy)
            epoch_val_top3_accuracies.append(val_top3_accuracy)

            # print results
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Epoch Train Loss: {train_loss:.4f} (took {train_time:.2f}s)")
            print(
                f"Epoch Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}, Top 3 Accuracy: {val_top3_accuracy:.2f} (took {val_time:.2f}s)"
            )

            # self.learning_rate_scheduler.step(val_loss)
            self.learning_rate_scheduler.step()

            save_start = time.perf_counter()
            self.save_model(
                epoch,
                epoch_train_losses,
                epoch_val_losses,
                epoch_val_accuracies,
                epoch_val_top3_accuracies,
            )
            save_time = time.perf_counter() - save_start

            epoch_time = time.perf_counter() - epoch_start
            print(f"Epoch completed in {epoch_time:.2f}s (Save: {save_time:.2f}s)")
            print("-" * 60)
        print(f"total training time: {(time.perf_counter()-start_time)//60}m")
        for i, train_loss in enumerate(epoch_train_losses):
            print(
                f"Epoch {i}: train loss={train_loss:.2f}, val_loss={epoch_val_losses[i]:.2f}, val accuracy={epoch_val_accuracies[i]:.2f}, val top 3 accuracy={epoch_val_accuracies[i]:.2f}"
            )
        return (
            epoch_train_losses,
            epoch_val_losses,
            epoch_val_accuracies,
            epoch_val_top3_accuracies,
        )

    def train_epoch(self):
        epoch_train_loss = 0
        total_batch_time = 0
        total_data_time = 0
        forward_time = 0
        backward_time = 0

        data_start = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            data_time = time.perf_counter() - data_start
            total_data_time += data_time

            batch_start = time.perf_counter()
            batch_train_loss, forward_time, backward_time = self.train_batch(
                inputs, targets
            )
            batch_time = time.perf_counter() - batch_start
            total_batch_time += batch_time

            epoch_train_loss += batch_train_loss
            if batch_idx % 50 == 0:
                print(
                    f"Batch {batch_idx}: Loss={batch_train_loss:.4f}, Time={batch_time:.3f}s"
                )
                print(f"  Data loading: {data_time:.3f}s")
                print(f"  Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")

            # memory cleanup
            del inputs, targets
            gc.collect()
            data_start = time.perf_counter()  # Start timing next data load

        avg_epoch_train_loss = epoch_train_loss / len(self.train_dataloader)
        print(f"Average batch time: {total_batch_time/len(self.train_dataloader):.3f}s")
        print(
            f"Average data loading time: {total_data_time/len(self.train_dataloader):.3f}s"
        )
        return avg_epoch_train_loss

    def train_batch(self, inputs, targets):
        self.optimizer.zero_grad()

        forward_start = time.perf_counter()
        preds = self.model.forward(inputs)
        loss = self.loss_fn(preds, targets)
        forward_time = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        loss.backward()
        self.optimizer.step()
        backward_time = time.perf_counter() - backward_start

        return loss.item(), forward_time, backward_time

    def run_model(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        correct_top_3 = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                preds = self.model.forward(inputs)
                loss = self.loss_fn(preds, targets)
                total_loss += loss.item()

                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == targets).sum().item()

                _, top3_indices = torch.topk(preds, k=3, dim=1)
                for i, target in enumerate(targets):
                    if target in top3_indices[i]:
                        correct_top_3 += 1

                total += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct / total) * 100
        top3_accuracy = (correct_top_3 / total) * 100

        self.model.train()
        return avg_loss, accuracy, top3_accuracy

    def save_model(
        self,
        epoch,
        epoch_train_losses,
        epoch_val_losses,
        epoch_val_accuracies,
        epoch_val_top3_accuracies,
    ):
        # if we haven't yet assigned it a save path
        if not self.save_dir:
            base_path = "models/"
            os.makedirs(base_path, exist_ok=True)
            i = 0
            while os.path.exists(os.path.join(base_path, f"model_{i}")):
                i += 1
            self.save_dir = os.path.join(base_path, f"model_{i}")
            os.makedirs(self.save_dir, exist_ok=True)
        model_dir = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        print(f"saving model checkpoint at {model_dir}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            "train_losses": epoch_train_losses,
            "val_losses": epoch_val_losses,
            "val_accuracies": epoch_val_accuracies,
            "val_top3_accuracies": epoch_val_top3_accuracies,
            "seed": self.seed,
            "labels": self.labels,
        }
        torch.save(checkpoint, model_dir)

    def load_checkpoint(self, checkpoint_path, seed):
        print(f"loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["seed"] != seed:
            raise ValueError(
                f"WARNING: different seed used in generating and splitting training data. Try again with correct seed: {checkpoint['seed']}"
            )
        # Load training state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.learning_rate_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return the checkpoint data
        return checkpoint
