"""Check held-out normalization, saved weights, and evaluation isolation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from jepa_policy import checkpoint_evaluation as entry
from jepa_policy.agent import TrainingAgent
from jepa_policy.datasets.icedtea_dataset import CAMERAS
from jepa_policy.flow_map import FlowMap
from jepa_policy.samplers import mip_sampler


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, observation, mask=None):
        return observation["agent_pos"].mean(dim=1) * self.scale

    def encode_rgb_features(self, observation):
        return observation["agent_pos"] * self.scale


class TinyJointNetwork(nn.Module):
    n_future_tokens = 1
    future_out_dim = 8

    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(-0.2))

    def joint_forward(self, action, start, end, observation, future):
        predicted = action * 0.1 + observation.unsqueeze(1) + self.marker
        return predicted, future * 0.1 + observation + self.marker


class RawEpisodeSource:
    normalize = False

    def __init__(self, root):
        self.root = root
        self.episodes = {1: {"dataset_from_index": 0, "dataset_to_index": 12}}
        self.raw_actions = np.repeat(
            np.linspace(10, 30, 12)[:, None], 8, axis=1
        ).astype(np.float32)
        self.raw_states = np.repeat(
            np.linspace(100, 120, 12)[:, None], 8, axis=1
        ).astype(np.float32)

    def __len__(self):
        return 3

    def read_image(self, record, camera, row):
        return torch.full((3, 4, 4), row, dtype=torch.uint8)

    def __getitem__(self, index):
        observation = {
            camera: torch.stack(
                [self.read_image({}, camera, row) for row in (index, index + 1)]
            )
            for camera in CAMERAS
        }
        future = {camera: self.read_image({}, camera, index + 5) for camera in CAMERAS}
        observation["agent_pos"] = torch.from_numpy(
            self.raw_states[index : index + 2].copy()
        )
        future["agent_pos"] = torch.from_numpy(self.raw_states[index + 5].copy())
        return {
            "obs": observation,
            "action": torch.from_numpy(self.raw_actions[index : index + 10].copy()),
            "future_obs": future,
        }


class MultipleEpisodeSource(RawEpisodeSource):
    def __init__(self, root, episode_count):
        super().__init__(root)
        self.episodes = {
            index + 1: {
                "dataset_from_index": index * 12,
                "dataset_to_index": (index + 1) * 12,
            }
            for index in range(episode_count)
        }
        self.raw_actions = np.tile(self.raw_actions, (episode_count, 1))
        self.raw_states = np.tile(self.raw_states, (episode_count, 1))
        self.raw_actions[-12:] += 1000

    def __len__(self):
        return len(self.episodes) * 3

    def __getitem__(self, index):
        episode, offset = divmod(index, 3)
        return super().__getitem__(episode * 12 + offset)


def make_agent():
    optimization = SimpleNamespace(
        device="cpu",
        future_joint_mode=True,
        use_future_embed_loss=True,
        future_embed_loss_mode="mip_two_step",
        future_head_only_stopgrad=False,
        t_two_step=0.9,
        future_t_two_step=0.9,
        loss_scale=1.0,
        future_state_loss_mode="ratio",
        future_state_loss_ratio=0.1,
        future_state_loss_weight_min=1e-6,
        future_state_loss_weight_max=0.1,
        loss_type="mip",
        ema_rate=0.995,
        num_steps=1,
    )
    agent = TrainingAgent.__new__(TrainingAgent)
    agent.config = SimpleNamespace(
        optimization=optimization,
        task=SimpleNamespace(obs_steps=2, horizon=10, act_dim=8),
    )
    agent.encoder = TinyEncoder()
    agent.flow_map = FlowMap(TinyJointNetwork())
    agent.encoder_ema = deepcopy(agent.encoder).requires_grad_(False)
    agent.flow_map_ema = deepcopy(agent.flow_map).requires_grad_(False)
    agent.sampler = mip_sampler
    agent._compiled_sampler = mip_sampler
    agent._future_nonfinite_logged = False
    return agent


@pytest.fixture
def context(tmp_path):
    agent = make_agent()
    training = SimpleNamespace(
        episodes={0: {}},
        action_min=np.full(8, 10, dtype=np.float32),
        action_range=np.full(8, 20, dtype=np.float32),
        state_min=np.full(8, 100, dtype=np.float32),
        state_range=np.full(8, 10, dtype=np.float32),
    )
    source = RawEpisodeSource(tmp_path)
    options = entry.CheckpointEvaluationOptions(tmp_path, batch_size=2)
    evaluator = entry.CheckpointEvaluator(agent.config, (training, source), options)
    statistics = {
        "schema_version": 1,
        "features": {
            "action": {
                "min": training.action_min.tolist(),
                "range": training.action_range.tolist(),
            },
            "observation.state": {
                "min": training.state_min.tolist(),
                "range": training.state_range.tolist(),
            },
        },
    }
    (tmp_path / "normalization_stats.json").write_text(json.dumps(statistics))
    checkpoint = tmp_path / "model_step_000001.pt"
    torch.save(
        {
            "encoder_ema": agent.encoder_ema.state_dict(),
            "flow_map_ema": agent.flow_map_ema.state_dict(),
            "training_state": {"n_gradient_step": 0},
        },
        checkpoint,
    )
    return agent, evaluator, checkpoint


def test_held_out_windows_reuse_training_ranges_without_clipping(context):
    _, evaluator, _ = context
    item = evaluator.windows[0]
    raw = evaluator.source[0]
    torch.testing.assert_close(item["action"], (raw["action"] - 10) / 20 * 2 - 1)
    torch.testing.assert_close(
        item["obs"]["agent_pos"], (raw["obs"]["agent_pos"] - 100) / 10 * 2 - 1
    )
    torch.testing.assert_close(
        item["future_obs"]["agent_pos"],
        (raw["future_obs"]["agent_pos"] - 100) / 10 * 2 - 1,
    )
    assert evaluator.states[-1, 0] == 3.0
    assert evaluator.source.raw_states[-1, 0] == 120.0


def test_saved_checkpoint_is_used_and_original_agent_is_untouched(context):
    agent, evaluator, checkpoint = context
    with torch.no_grad():
        agent.flow_map_ema.net.marker.fill_(9.0)
    agent.encoder.train()
    agent.flow_map.train()
    original_parameter = agent.flow_map_ema.net.marker.detach().clone()
    saved = evaluator.load_saved_agent(agent, checkpoint, 1)
    assert saved.flow_map.net.marker.item() == pytest.approx(-0.2)
    assert (
        saved.flow_map.net.marker.data_ptr() != agent.flow_map_ema.net.marker.data_ptr()
    )
    assert saved.encoder is saved.encoder_ema
    assert not saved.encoder.training and not saved.flow_map.training
    torch.testing.assert_close(agent.flow_map_ema.net.marker, original_parameter)
    assert agent.encoder.training and agent.flow_map.training


def test_validation_uses_real_training_loss_and_weights_partial_batch(context):
    agent, evaluator, checkpoint = context
    saved = evaluator.load_saved_agent(agent, checkpoint, 1)
    actual = evaluator.validation_loss(saved)
    records = []
    for index in range(2):
        batch = torch.utils.data.default_collate(
            [evaluator.windows[row] for row in range(index * 2, min(index * 2 + 2, 3))]
        )
        batch = entry.move_evaluation_batch(batch, torch.device("cpu"))
        records.append(
            saved.validation_metrics(
                batch["action"],
                batch["obs"],
                torch.ones(len(batch["action"])),
                batch["future_obs"],
                seed=evaluator.options.seed + index,
            )
        )
    for name in ("loss_total", "loss_action", "loss_future_raw"):
        assert actual[name] == pytest.approx(
            (records[0][name] * 2 + records[1][name]) / 3
        )
    assert actual["windows"] == 3
    assert actual["loss"] == actual["loss_total"]


def test_evaluation_saves_dense_curves_and_restores_rng(context, monkeypatch):
    agent, evaluator, checkpoint = context
    rendered = []
    monkeypatch.setattr(
        entry, "render_episode", lambda *args, **kwargs: rendered.append(args)
    )
    agent.encoder.train()
    random_state = torch.get_rng_state().clone()
    result = evaluator.evaluate(agent, checkpoint, 1)
    assert torch.equal(random_state, torch.get_rng_state())
    assert agent.encoder.training
    assert result["frames"] == 12 and result["windows"] == 3 and result["episodes"] == 1
    assert result["loss"] != pytest.approx(result["shadow_action_mse"])
    output = checkpoint.parent / (checkpoint.stem + "_eval")
    assert (output / "metrics.json").is_file()
    metadata = json.loads((output / "metrics.json").read_text())
    statistics_path = checkpoint.parent / "normalization_stats.json"
    assert (
        metadata["normalization"]["sha256"]
        == hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    )
    assert metadata["evaluation_episode_indices"] == [1]
    assert metadata["training_episode_indices"] == [0]
    assert "jepa_policy/agent.py" in metadata["source_hashes"]
    assert len(rendered) == 2
    with np.load(output / "artifacts/episode_000001.npz") as artifact:
        np.testing.assert_array_equal(
            artifact["ground_truth"], evaluator.source.raw_actions
        )
        assert artifact["prediction"].shape == (12, 8)
    with pytest.raises(FileExistsError):
        evaluator.evaluate(agent, checkpoint, 1)


def test_checkpoint_step_mismatch_fails(context):
    agent, evaluator, checkpoint = context
    with pytest.raises(ValueError, match="step"):
        evaluator.load_saved_agent(agent, checkpoint, 2)


def test_evaluation_failure_restores_random_state(context, monkeypatch):
    agent, evaluator, checkpoint = context

    def fail(*args):
        torch.rand(5)
        raise ValueError("expected validation failure")

    monkeypatch.setattr(evaluator, "validation_loss", fail)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="expected"):
        evaluator.evaluate(agent, checkpoint, 1)
    assert torch.equal(before, torch.get_rng_state())
    assert agent.encoder.training


def test_evaluation_rejects_statistics_file_different_from_training(context):
    agent, evaluator, checkpoint = context
    path = checkpoint.parent / "normalization_stats.json"
    statistics = json.loads(path.read_text())
    statistics["features"]["action"]["min"][0] += 1.0
    path.write_text(json.dumps(statistics))
    with pytest.raises(ValueError, match="normalization differs"):
        evaluator.evaluate(agent, checkpoint, 1)


@pytest.mark.parametrize("episode_count", [1, 2, 3, 10])
def test_all_episodes_contribute_to_metrics_but_only_first_three_are_plotted(
    context, monkeypatch, episode_count
):
    agent, previous, checkpoint = context
    source = MultipleEpisodeSource(checkpoint.parent, episode_count)
    evaluator = entry.CheckpointEvaluator(
        agent.config, (previous.training, source), previous.options
    )
    rendered = []
    monkeypatch.setattr(
        entry,
        "render_episode",
        lambda path, *args, **kwargs: rendered.append(Path(path).stem),
    )
    metrics = evaluator.evaluate(agent, checkpoint, 1)
    output = checkpoint.parent / (checkpoint.stem + "_eval")
    artifacts = sorted((output / "artifacts").glob("*.npz"))
    assert len(artifacts) == episode_count
    assert metrics["episodes"] == episode_count
    assert metrics["frames"] == episode_count * 12
    assert metrics["windows"] == episode_count * 3
    plotted = [f"episode_{index:06d}" for index in range(1, min(episode_count, 3) + 1)]
    assert rendered == [episode for episode in plotted for _ in range(2)]
    metadata = json.loads((output / "metrics.json").read_text())
    assert metadata["plotted_episode_ids"] == plotted
    assert metadata["evaluation_episode_indices"] == list(range(1, episode_count + 1))
    squared_errors = []
    for path in artifacts:
        with np.load(path) as artifact:
            squared_errors.append(
                np.square(
                    artifact["prediction"].astype(np.float64)
                    - artifact["ground_truth"].astype(np.float64)
                )
            )
    expected = np.concatenate(squared_errors).mean(axis=0)
    assert metrics["joint_mse_sum"] == pytest.approx(expected[:7].sum())
    assert metrics["gripper_mse"] == pytest.approx(expected[7])
    if episode_count > 3:
        assert (
            metrics["gripper_mse"] > np.concatenate(squared_errors[:3]).mean(axis=0)[7]
        )
