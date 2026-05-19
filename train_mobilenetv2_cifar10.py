import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a MobileNetV2 transfer-learning model on CIFAR-10 or run inference."
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("saved_model/mobilenetv2_cifar10"))
    parser.add_argument("--infer-image", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap to speed up local checks.",
    )
    return parser.parse_args()


def preprocess_image(image: tf.Tensor, img_size: int) -> tf.Tensor:
    image = tf.image.resize(image, (img_size, img_size))
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def build_datasets(
    img_size: int,
    batch_size: int,
    validation_split: float,
    max_train_samples: int | None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    if max_train_samples is not None:
        x_train = x_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    val_size = int(len(x_train) * validation_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    def _train_map(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        image = data_augmentation(image)
        return preprocess_image(image, img_size), label

    def _eval_map(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        return preprocess_image(image, img_size), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(10000, len(x_train)))
        .map(_train_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(_eval_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(_eval_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, x_test, y_test


def build_model(img_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    metrics = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics.get("loss", []), label="train_loss")
    plt.plot(metrics.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(metrics.get("accuracy", []), label="train_acc")
    plt.plot(metrics.get("val_accuracy", []), label="val_acc")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()
    summary: Dict[str, Dict[str, float]] = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        summary[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return summary


def train_and_evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, x_test, y_test = build_datasets(
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        max_train_samples=args.max_train_samples,
    )

    model = build_model(args.img_size)

    warmup_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    if args.fine_tune_epochs > 0:
        base_model = model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
        )

        for key, values in fine_tune_history.history.items():
            warmup_history.history.setdefault(key, []).extend(values)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    predictions = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "per_class": classification_metrics(y_test, y_pred),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(warmup_history, args.output_dir / "training_curves.png")
    model.save(args.model_path)

    sample_pred = np.argmax(
        model.predict(
            tf.expand_dims(preprocess_image(tf.cast(x_test[0], tf.float32), args.img_size), axis=0),
            verbose=0,
        )[0]
    )
    print(f"Sample prediction for test image 0: {CLASS_NAMES[sample_pred]}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved curves: {args.output_dir / 'training_curves.png'}")


def infer(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Run training first or provide --model-path."
        )

    model = tf.keras.models.load_model(args.model_path)

    if args.infer_image is not None:
        image_bytes = tf.io.read_file(str(args.infer_image))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        source = str(args.infer_image)
    else:
        (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        image = tf.convert_to_tensor(x_test[args.sample_index])
        source = f"CIFAR-10 test sample {args.sample_index}"

    image = tf.cast(image, tf.float32)
    input_tensor = tf.expand_dims(preprocess_image(image, args.img_size), axis=0)

    probs = model.predict(input_tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    print(f"Inference source: {source}")
    print(f"Predicted class: {CLASS_NAMES[top_idx]} ({probs[top_idx]:.4f})")


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train_and_evaluate(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
