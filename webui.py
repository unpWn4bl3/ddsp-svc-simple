import os
import shutil
import threading
from pathlib import Path

import gradio as gr
import soundfile as sf
import io

from loguru import logger

from ddsp_svc.config import load_config, save_config
from ddsp_svc.device import detect_device
from ddsp_svc.infer import InferencePipeline
from ddsp_svc.preprocess import run_preprocessing

CONFIG_PATH = "configs/default.yaml"


def get_config():
    if not os.path.exists(CONFIG_PATH):
        from ddsp_svc.config import SvcConfig
        cfg = SvcConfig()
        save_config(cfg, CONFIG_PATH)
    return load_config(CONFIG_PATH)


def get_raw_speakers():
    raw = Path("data/raw")
    if not raw.exists():
        return "（data/raw/ 不存在）"
    dirs = sorted(d for d in raw.iterdir() if d.is_dir() and any(d.rglob("*.wav")))
    if not dirs:
        return "（data/raw/ 下没有带 wav 的子文件夹）"
    return ", ".join(d.name for d in dirs)


def on_preprocess(f0_extractor, encoder, device_choice, progress=gr.Progress()):
    cfg = get_config()
    cfg.data.f0_extractor = f0_extractor
    cfg.data.encoder = encoder
    cfg.device = device_choice

    buf = io.StringIO()
    sink_id = logger.add(buf, format="{message}", level="INFO")
    try:
        progress(0, desc="预处理...")
        run_preprocessing(cfg)
        progress(1.0, desc="完成")
    finally:
        logger.remove(sink_id)
    return buf.getvalue(), get_raw_speakers()


def on_train(batch_size, learning_rate, epochs, save_interval, device_choice, progress=gr.Progress()):
    cfg = get_config()
    cfg.train.batch_size = batch_size
    cfg.train.learning_rate = learning_rate
    cfg.train.epochs = epochs
    cfg.train.interval_val = save_interval
    cfg.device = device_choice
    save_config(cfg, CONFIG_PATH)

    def train_worker():
        from ddsp_svc.train import Trainer
        trainer = Trainer(cfg)
        trainer.run()

    thread = threading.Thread(target=train_worker, daemon=True)
    thread.start()
    gr.Info("Training started in background. Check terminal for logs.")
    return "⏳ Training running..."


def on_stop_training():
    cfg = get_config()
    stop_file = Path(cfg.env.expdir) / "stop.txt"
    stop_file.touch()
    gr.Info("Stop signal sent")
    return "🛑 Stopping..."


def on_archive(archive_name):
    if not archive_name:
        raise gr.Error("请输入归档名称")
    src_dir = Path("exp/default")
    dst_dir = Path("exp") / archive_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(src_dir.glob("model_*.pt"))
    if not ckpts:
        raise gr.Error(f"{src_dir}/ 下没有 checkpoint 可归档")
    for c in ckpts:
        dst = dst_dir / c.name
        if not dst.exists():
            shutil.copy2(c, dst)
    names = ", ".join(c.name for c in ckpts)
    gr.Info(f"已归档到 {dst_dir}/，共 {len(ckpts)} 个文件")
    return names


def list_checkpoints():
    exp_root = Path("exp")
    ckpts = []
    for d in sorted(exp_root.iterdir()):
        if d.is_dir():
            ckpts.extend(sorted(d.glob("model_*.pt")))
    return [str(c) for c in ckpts] if ckpts else ["No checkpoints found"]


def on_infer(audio_input, checkpoint, speaker, keychange, infer_step, method, t_start, progress=gr.Progress()):
    if not audio_input:
        raise gr.Error("Please upload an audio file")

    cfg = get_config()
    pipeline = InferencePipeline(cfg)
    pipeline.load_model(checkpoint)

    output = pipeline.infer(
        audio_input, speaker=speaker, keychange=keychange,
        infer_step=infer_step, method=method, t_start=t_start,
    )
    out_path = "tmp/output.wav"
    os.makedirs("tmp", exist_ok=True)
    sf.write(out_path, output, cfg.data.sampling_rate)
    return out_path


def create_ui():
    cfg = get_config()
    device_choices = ["auto", "cuda", "mps", "cpu"]
    available = [d for d in device_choices if d == "auto" or
                 (d == "cuda" and __import__("torch").cuda.is_available()) or
                 (d == "mps" and __import__("torch").backends.mps.is_available()) or
                 d == "cpu"]

    with gr.Blocks(title="DDSP-SVC") as app:
        gr.Markdown("# DDSP-SVC")

        with gr.Tab("① 数据准备"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 从 `data/raw/` 读取 wav（子文件夹名为说话人）")
                    speaker_display = gr.Textbox(
                        value=get_raw_speakers(), label="检测到的说话人",
                        max_lines=3, interactive=False,
                    )
                    f0_choice = gr.Dropdown(
                        choices=["parselmouth", "crepe", "rmvpe"],
                        value=cfg.data.f0_extractor, label="F0 提取器",
                    )
                    enc_choice = gr.Dropdown(
                        choices=["contentvec768l12tta2x", "contentvec768l12"],
                        value=cfg.data.encoder, label="编码器",
                    )
                    dev_choice = gr.Dropdown(
                        choices=available, value=cfg.device, label="设备",
                    )
                    preprocess_btn = gr.Button("开始预处理", variant="primary")
                with gr.Column():
                    detail_box = gr.Textbox(label="详细信息", lines=15, max_lines=30, interactive=False)

            preprocess_btn.click(
                on_preprocess, [f0_choice, enc_choice, dev_choice], [detail_box, speaker_display],
            )

        with gr.Tab("② 训练"):
            with gr.Row():
                with gr.Column():
                    bs = gr.Number(value=cfg.train.batch_size, label="Batch Size", precision=0)
                    lr = gr.Number(value=cfg.train.learning_rate, label="Learning Rate")
                    ep = gr.Number(value=cfg.train.epochs, label="Epochs", precision=0)
                    save_int = gr.Number(value=cfg.train.interval_val, label="保存间隔（步数）", precision=0)
                    dev = gr.Dropdown(choices=available, value=cfg.device, label="设备")
                with gr.Column():
                    train_status = gr.Textbox(label="状态", interactive=False)
                    train_btn = gr.Button("开始训练", variant="primary")
                    stop_btn = gr.Button("停止训练", variant="stop")

            train_btn.click(on_train, [bs, lr, ep, save_int, dev], train_status)
            stop_btn.click(on_stop_training, None, train_status)

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    archive_name = gr.Textbox(label="归档名称（将 exp/default/ 下的模型复制到 exp/名称/）")
                    archive_btn = gr.Button("归档模型", variant="secondary")
                with gr.Column():
                    archive_result = gr.Textbox(label="归档结果", interactive=False)

            archive_btn.click(on_archive, archive_name, archive_result)

        with gr.Tab("③ 推理"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(label="输入音频", type="filepath")
                    ckpt = gr.Dropdown(choices=list_checkpoints(), label="模型检查点")
                    refresh_btn = gr.Button("刷新模型列表")
                    refresh_btn.click(
                        lambda: gr.Dropdown(choices=list_checkpoints()),
                        None, ckpt,
                    )
                    spk = gr.Dropdown(choices=cfg.spks, value=cfg.spks[0] if cfg.spks else "speaker0", label="说话人")
                    key = gr.Slider(-12, 12, 0, step=1, label="变调 (半音)")
                with gr.Column():
                    steps = gr.Slider(1, 100, cfg.infer.infer_step, step=1, label="推理步数")
                    method = gr.Dropdown(["euler", "rk4"], value=cfg.infer.method, label="采样方法")
                    t_start = gr.Slider(0.0, 1.0, cfg.model.t_start, step=0.1, label="起始时间")
                    audio_out = gr.Audio(label="输出音频", type="filepath")
                    infer_btn = gr.Button("开始推理", variant="primary")

            infer_btn.click(
                on_infer, [audio_in, ckpt, spk, key, steps, method, t_start], audio_out,
            )

    return app


app = create_ui()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, css="footer {display:none !important}")
