import matplotlib.pyplot as plt
import numpy as np

class MetricTracker:
    def __init__(self):
        self.epochs = []
        self.sensitivity = []
        self.specificity = []
        self.f1_score = []
        self.accuracy = []
        self.mean_iou = []
        self.mean_loss = []
        self.current_epoch = 0

    def update(self, se, sp, f1, acc, miou, loss):
        self.current_epoch += 1
        self.epochs.append(self.current_epoch)
        self.sensitivity.append(se)
        self.specificity.append(sp)
        self.f1_score.append(f1)
        self.accuracy.append(acc)
        self.mean_iou.append(miou)
        self.mean_loss.append(loss)

    def plot_metrics_and_loss(self, save_path=None):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        def plot(ax, data, label, color):
            x = range(1, len(data) + 1)
            ax.plot(x, data, color, label=label)

            # 动态设置 y 轴范围
            y_min, y_max = min(data), max(data)
            margin = (y_max - y_min) * 0.1  # 增加 10% 的上下边距
            ax.set_ylim(y_min - margin, y_max + margin)

            # 固定 y 轴刻度数
            num_ticks = 5  # 设定刻度数量
            y_ticks = np.linspace(y_min, y_max, num_ticks)
            ax.set_yticks(y_ticks)

            ax.set_xlabel('Epochs')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid()

        # 绘制指标曲线
        plot(axs[0, 0], self.sensitivity, 'Sensitivity', 'b-')
        plot(axs[0, 1], self.specificity, 'Specificity', 'g-')
        plot(axs[1, 0], self.f1_score, 'F1-Score', 'c-')
        plot(axs[1, 1], self.accuracy, 'Accuracy', 'm-')
        plot(axs[2, 0], self.mean_iou, 'Mean IoU', 'y-')
        plot(axs[2, 1], self.mean_loss, 'Mean Loss', 'r-')

        fig.suptitle('Training Metrics Over Epochs (Smoothed)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved at: {save_path}")

        plt.show()

# 使用示例
if __name__ == "__main__":
    tracker = MetricTracker()
    tracker.update(0.8, 0.9, 0.85, 0.88, 0.75, 0.3)
    tracker.update(0.82, 0.91, 0.87, 0.89, 0.78, 0.25)
    tracker.update(0.85, 0.93, 0.89, 0.9, 0.8, 0.2)
    tracker.update(0.87, 0.94, 0.9, 0.91, 0.82, 0.18)
    tracker.update(0.88, 0.95, 0.91, 0.92, 0.83, 0.15)
    tracker.update(0.9, 0.9, 0.88, 0.9, 0.82, 0.24)
    tracker.update(0.93, 0.91, 0.89, 0.88, 0.77, 0.2)
    tracker.update(0.8, 0.92, 0.91, 0.92, 0.77, 0.22)
    tracker.update(0.92, 0.9, 0.93, 0.94, 0.78, 0.13)
    tracker.update(0.94, 0.93, 0.86, 0.89, 0.83, 0.22)
    tracker.update(0.92, 0.97, 0.9, 0.96, 0.79, 0.21)
    tracker.update(0.92, 0.96, 0.94, 0.93, 0.82, 0.11)
    tracker.update(0.83, 0.93, 0.86, 0.9, 0.76, 0.16)
    tracker.update(0.9, 0.93, 0.89, 0.9, 0.78, 0.29)
    tracker.update(0.9, 0.95, 0.87, 0.94, 0.77, 0.18)
    tracker.update(0.95, 0.96, 0.91, 0.93, 0.83, 0.26)
    tracker.update(0.83, 0.9, 0.88, 0.9, 0.77, 0.29)
    tracker.update(0.93, 0.93, 0.92, 0.91, 0.84, 0.19)
    tracker.update(0.84, 0.92, 0.91, 0.9, 0.81, 0.28)
    tracker.update(0.86, 0.92, 0.95, 0.92, 0.76, 0.11)
    tracker.update(0.82, 0.96, 0.93, 0.91, 0.76, 0.18)
    tracker.update(0.95, 0.95, 0.95, 0.95, 0.75, 0.24)
    tracker.update(0.9, 0.95, 0.88, 0.93, 0.76, 0.19)
    tracker.update(0.87, 0.99, 0.94, 0.9, 0.8, 0.14)
    tracker.update(0.94, 0.98, 0.88, 0.93, 0.81, 0.13)
    tracker.update(0.8, 0.9, 0.85, 0.88, 0.75, 0.3)
    tracker.update(0.82, 0.91, 0.87, 0.89, 0.78, 0.25)
    tracker.update(0.85, 0.93, 0.89, 0.9, 0.8, 0.2)
    tracker.update(0.87, 0.94, 0.9, 0.91, 0.82, 0.18)
    tracker.update(0.88, 0.95, 0.91, 0.92, 0.83, 0.15)
    tracker.update(0.9, 0.9, 0.88, 0.9, 0.82, 0.24)
    tracker.update(0.93, 0.91, 0.89, 0.88, 0.77, 0.2)
    tracker.update(0.8, 0.92, 0.91, 0.92, 0.77, 0.22)
    tracker.update(0.92, 0.9, 0.93, 0.94, 0.78, 0.13)
    tracker.update(0.94, 0.93, 0.86, 0.89, 0.83, 0.22)
    tracker.update(0.92, 0.97, 0.9, 0.96, 0.79, 0.21)
    tracker.update(0.92, 0.96, 0.94, 0.93, 0.82, 0.11)
    tracker.update(0.83, 0.93, 0.86, 0.9, 0.76, 0.16)
    tracker.update(0.9, 0.93, 0.89, 0.9, 0.78, 0.29)
    tracker.update(0.9, 0.95, 0.87, 0.94, 0.77, 0.18)
    tracker.update(0.95, 0.96, 0.91, 0.93, 0.83, 0.26)
    tracker.update(0.83, 0.9, 0.88, 0.9, 0.77, 0.29)
    tracker.update(0.93, 0.93, 0.92, 0.91, 0.84, 0.19)
    tracker.update(0.84, 0.92, 0.91, 0.9, 0.81, 0.28)
    tracker.update(0.86, 0.92, 0.95, 0.92, 0.76, 0.11)
    tracker.update(0.82, 0.96, 0.93, 0.91, 0.76, 0.18)
    tracker.update(0.95, 0.95, 0.95, 0.95, 0.75, 0.24)
    tracker.update(0.9, 0.95, 0.88, 0.93, 0.76, 0.19)
    tracker.update(0.87, 0.99, 0.94, 0.9, 0.8, 0.14)
    tracker.update(0.94, 0.98, 0.88, 0.93, 0.81, 0.13)

    tracker.plot_metrics_and_loss(save_path='training_metrics_over_epochs.png')
