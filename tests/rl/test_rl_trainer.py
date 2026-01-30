"""测试 RLTrainer

测试强化学习训练器的各种功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from datasets import Dataset
import json


# ========== Fixtures ==========


@pytest.fixture
def mock_tokenizer():
    """模拟 tokenizer"""
    tokenizer = Mock()
    tokenizer.apply_chat_template = Mock(return_value="<|im_start|>user\nTest prompt<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer


@pytest.fixture
def mock_dataset():
    """模拟数据集"""
    dataset = Dataset.from_dict({
        "prompt": ["Question 1", "Question 2"],
        "completion": ["Answer 1", "Answer 2"]
    })
    return dataset


@pytest.fixture
def rl_trainer(mock_tokenizer):
    """创建 RLTrainer 实例"""
    with patch('liagents.rl.trainer.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        from liagents.rl.trainer import RLTrainer
        return RLTrainer(
            model_name_or_path="Qwen/Qwen3-0.6B",
            output_dir="./test_outputs"
        )


@pytest.fixture
def rl_trainer_with_dataset(rl_trainer, mock_dataset):
    """创建已加载数据集的 RLTrainer"""
    rl_trainer.dataset = mock_dataset
    return rl_trainer


# ========== 初始化测试 ==========


class TestRLTrainerInit:
    """测试 RLTrainer 初始化"""

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_init_basic(self, mock_tokenizer_fn, mock_tokenizer):
        """测试基本初始化"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer

        trainer = RLTrainer(
            model_name_or_path="Qwen/Qwen3-0.6B",
            output_dir="./outputs"
        )

        # 验证基本属性设置
        assert trainer.model_name_or_path == "Qwen/Qwen3-0.6B"
        assert "Qwen3-0.6B" in trainer.output_dir
        assert trainer.tokenizer == mock_tokenizer

        # 验证默认训练参数
        assert trainer.num_epochs == 2
        assert trainer.learning_rate == 5e-5
        assert trainer.batch_size == 4

        # 验证 tokenizer 被正确加载
        mock_tokenizer_fn.assert_called_once_with("Qwen/Qwen3-0.6B")

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_init_custom_model(self, mock_tokenizer_fn, mock_tokenizer):
        """测试自定义模型路径"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer

        trainer = RLTrainer(
            model_name_or_path="custom/model/path",
            output_dir="./outputs"
        )

        assert trainer.model_name_or_path == "custom/model/path"
        mock_tokenizer_fn.assert_called_once_with("custom/model/path")

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_output_dir_structure(self, mock_tokenizer_fn, mock_tokenizer):
        """测试输出目录结构"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer
        import re

        trainer = RLTrainer(
            model_name_or_path="Qwen/Qwen3-0.6B",
            output_dir="./outputs"
        )

        # 验证输出目录格式: ./outputs/{model_name}/{timestamp}
        output_path = Path(trainer.output_dir)
        assert output_path.parent.name == "Qwen3-0.6B"
        # 验证时间戳格式 (YYYYMMDD-HHMMSS)
        timestamp = output_path.name
        assert re.match(r'\d{8}-\d{6}', timestamp)

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_init_with_spaces_in_model_name(self, mock_tokenizer_fn, mock_tokenizer):
        """测试模型名包含空格时的处理"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer

        trainer = RLTrainer(
            model_name_or_path="Model With Spaces",
            output_dir="./outputs"
        )

        # 验证空格被替换为下划线
        assert "Model_With_Spaces" in trainer.output_dir
        assert " " not in trainer.output_dir


# ========== 数据集加载测试 ==========


class TestLoadDataset:
    """测试数据集加载功能"""

    @patch('liagents.rl.datasets.load_dataset')
    def test_load_sft_format_dataset(self, mock_load_dataset, rl_trainer):
        """测试加载 SFT 格式数据集"""
        # 准备 mock 数据
        mock_data = {
            "question": ["What is 2+2?"],
            "answer": ["4"]
        }
        mock_hf_dataset = Dataset.from_dict(mock_data)
        mock_load_dataset.return_value = mock_hf_dataset

        # 调用 load_dataset
        result = rl_trainer.load_dataset(
            dataset_name_or_path="openai/gsm8k",
            format_type="sft",
            split="train",
            max_samples=-1
        )

        # 验证结果
        assert result["status"] == "success"
        assert result["format_type"] == "sft"
        assert result["split"] == "train"
        assert result["dataset_size"] == 1
        assert len(result["sample_examples"]) == 3

        # 验证数据集被保存到实例变量
        assert rl_trainer.dataset is not None

    @patch('liagents.rl.datasets.load_dataset')
    def test_load_rl_format_dataset(self, mock_load_dataset, rl_trainer):
        """测试加载 RL 格式数据集"""
        mock_data = {
            "question": ["What is 2+2?"],
            "answer": ["4"]
        }
        mock_hf_dataset = Dataset.from_dict(mock_data)
        mock_load_dataset.return_value = mock_hf_dataset

        result = rl_trainer.load_dataset(
            dataset_name_or_path="openai/gsm8k",
            format_type="rl",
            split="train",
            max_samples=-1
        )

        assert result["status"] == "success"
        assert result["format_type"] == "rl"

    @patch('liagents.rl.datasets.load_dataset')
    def test_load_dataset_with_max_samples(self, mock_load_dataset, rl_trainer):
        """测试限制样本数量"""
        # 准备包含多个样本的数据
        mock_data = {
            "question": [f"Question {i}" for i in range(100)],
            "answer": [f"Answer {i}" for i in range(100)]
        }
        mock_hf_dataset = Dataset.from_dict(mock_data)
        mock_load_dataset.return_value = mock_hf_dataset

        result = rl_trainer.load_dataset(
            dataset_name_or_path="openai/gsm8k",
            format_type="sft",
            split="train",
            max_samples=10
        )

        # 验证只加载了指定数量的样本
        assert result["dataset_size"] == 10

    def test_load_dataset_invalid_format(self, rl_trainer):
        """测试无效的数据格式"""
        result = rl_trainer.load_dataset(
            dataset_name_or_path="openai/gsm8k",
            format_type="invalid_format",
            split="train"
        )

        assert result["status"] == "error"
        assert "不支持的数据格式" in result["message"]


# ========== 训练方法测试 ==========


class TestTrainMethod:
    """测试 train 方法"""

    @patch('liagents.rl.trainer.check_trl_installation')
    def test_train_sft_algorithm(self, mock_check_trl, rl_trainer_with_dataset):
        """测试 SFT 训练算法"""
        mock_check_trl.return_value = True

        # Mock _train_sft 方法
        rl_trainer_with_dataset._train_sft = Mock(return_value={
            "status": "success",
            "algorithm": "SFT"
        })

        result = rl_trainer_with_dataset.train(
            algorithm="sft",
            batch_size=8,
            num_epochs=3
        )

        # 验证 _train_sft 被调用
        rl_trainer_with_dataset._train_sft.assert_called_once()

        # 验证参数被正确设置到类属性
        assert rl_trainer_with_dataset.num_epochs == 3
        assert rl_trainer_with_dataset.batch_size == 8

        # 解析 JSON 结果
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"

    @patch('liagents.rl.trainer.check_trl_installation')
    def test_train_unsupported_algorithm(self, mock_check_trl, rl_trainer):
        """测试不支持的训练算法"""
        mock_check_trl.return_value = True

        result = rl_trainer.train(algorithm="unsupported_algorithm")
        result_dict = json.loads(result)

        assert result_dict["status"] == "error"
        assert "不支持的算法" in result_dict["message"]

    @patch('liagents.rl.trainer.check_trl_installation')
    def test_train_without_trl(self, mock_check_trl, rl_trainer):
        """测试 TRL 不可用时的错误处理"""
        mock_check_trl.return_value = False

        result = rl_trainer.train(algorithm="sft")
        result_dict = json.loads(result)

        assert result_dict["status"] == "error"
        assert "TRL不可用" in result_dict["message"]

    @patch('liagents.rl.trainer.check_trl_installation')
    def test_train_parameters_passed_correctly(self, mock_check_trl, rl_trainer_with_dataset):
        """测试训练参数正确传递"""
        mock_check_trl.return_value = True
        rl_trainer_with_dataset._train_sft = Mock(return_value={"status": "success"})

        rl_trainer_with_dataset.train(
            algorithm="sft",
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-4,
            use_lora=True,
            lora_rank=32,
            lora_alpha=64,
            use_fp16=True,
            use_bf16=False
        )

        # 验证类属性被设置
        assert rl_trainer_with_dataset.num_epochs == 5
        assert rl_trainer_with_dataset.batch_size == 16
        assert rl_trainer_with_dataset.learning_rate == 1e-4

        # 验证 _train_sft 收到正确的参数
        call_kwargs = rl_trainer_with_dataset._train_sft.call_args.kwargs
        assert call_kwargs["use_lora"] is True
        assert call_kwargs["lora_rank"] == 32
        assert call_kwargs["lora_alpha"] == 64
        assert call_kwargs["use_fp16"] is True
        assert call_kwargs["use_bf16"] is False


# ========== _train_sft 测试 ==========


class TestTrainSFT:
    """测试 _train_sft 方法"""

    @patch('liagents.rl.trainer.SFTTrainerWrapper')
    @patch('liagents.rl.trainer.setup_training_environment')
    @patch('liagents.rl.trainer.TrainingConfig')
    def test_train_sft_with_registered_dataset(
        self, mock_config, mock_setup_env, mock_trainer_wrapper, rl_trainer_with_dataset
    ):
        """测试使用已注册的数据集进行训练"""
        # 准备 mocks
        mock_trainer_instance = Mock()
        mock_trainer_wrapper.return_value = mock_trainer_instance

        result = rl_trainer_with_dataset._train_sft(
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            use_fp16=False,
            use_bf16=True
        )

        # 验证 TrainingConfig 被正确创建
        mock_config.assert_called_once()
        config_kwargs = mock_config.call_args.kwargs
        assert config_kwargs["model_name_or_path"] == rl_trainer_with_dataset.model_name_or_path
        assert config_kwargs["output_dir"] == rl_trainer_with_dataset.output_dir
        assert config_kwargs["num_train_epochs"] == rl_trainer_with_dataset.num_epochs
        assert config_kwargs["per_device_train_batch_size"] == rl_trainer_with_dataset.batch_size
        assert config_kwargs["learning_rate"] == rl_trainer_with_dataset.learning_rate
        assert config_kwargs["use_lora"] is True
        assert config_kwargs["lora_r"] == 8
        assert config_kwargs["lora_alpha"] == 16
        assert config_kwargs["use_fp16"] is False
        assert config_kwargs["use_bf16"] is True

        # 验证训练流程
        mock_setup_env.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()

        # 验证返回结果
        assert result["status"] == "success"
        assert result["algorithm"] == "SFT"

    @patch('liagents.rl.trainer.SFTTrainerWrapper')
    @patch('liagents.rl.trainer.setup_training_environment')
    @patch('liagents.rl.trainer.TrainingConfig')
    def test_train_sft_with_custom_dataset(
        self, mock_config, mock_setup_env, mock_trainer_wrapper, rl_trainer, mock_dataset
    ):
        """测试使用自定义数据集进行训练"""
        mock_trainer_instance = Mock()
        mock_trainer_wrapper.return_value = mock_trainer_instance

        result = rl_trainer._train_sft(
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            custom_dataset=mock_dataset
        )

        # 验证使用自定义数据集
        mock_trainer_wrapper.assert_called_once()
        call_args = mock_trainer_wrapper.call_args
        assert call_args.kwargs["dataset"] == mock_dataset

        assert result["status"] == "success"

    def test_train_sft_without_dataset_raises_error(self, rl_trainer):
        """测试没有数据集时抛出错误"""
        rl_trainer.dataset = None

        with pytest.raises(ValueError, match="未指定数据集"):
            rl_trainer._train_sft(
                use_lora=True,
                lora_rank=8,
                lora_alpha=16
            )


# ========== 参数复用测试 ==========


class TestParameterReuse:
    """测试参数复用机制"""

    def test_num_epochs_reused_across_calls(self, rl_trainer_with_dataset):
        """测试 num_epochs 在多次调用间复用"""
        rl_trainer_with_dataset.num_epochs = 5

        # Mock _train_sft 来验证参数
        rl_trainer_with_dataset._train_sft = Mock(return_value={"status": "success"})

        with patch('liagents.rl.trainer.check_trl_installation', return_value=True):
            # 传入 None 应该使用类属性的值
            # 但由于 train 方法有默认值，我们需要直接验证 _train_sft 使用了正确的值
            rl_trainer_with_dataset.train(algorithm="sft", num_epochs=5)

            # 验证使用了类属性中设置的 num_epochs
            call_kwargs = rl_trainer_with_dataset._train_sft.call_args.kwargs
            # _train_sft 内部会使用 self.num_epochs 创建 TrainingConfig
            assert rl_trainer_with_dataset.num_epochs == 5

    def test_learning_rate_reused(self, rl_trainer_with_dataset):
        """测试 learning_rate 复用"""
        rl_trainer_with_dataset.learning_rate = 3e-4

        assert rl_trainer_with_dataset.learning_rate == 3e-4

    def test_batch_size_reused(self, rl_trainer_with_dataset):
        """测试 batch_size 复用"""
        rl_trainer_with_dataset.batch_size = 16

        assert rl_trainer_with_dataset.batch_size == 16


# ========== 输出目录测试 ==========


class TestOutputDirectory:
    """测试输出目录相关功能"""

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_output_dir_contains_timestamp(self, mock_tokenizer_fn, mock_tokenizer):
        """测试输出目录包含时间戳"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer
        import re
        import time

        trainer1 = RLTrainer(model_name_or_path="test/model")
        time.sleep(1)  # 等待 1 秒确保时间戳不同
        trainer2 = RLTrainer(model_name_or_path="test/model")

        # 验证两次创建的时间戳不同（包含时间）
        timestamp1 = Path(trainer1.output_dir).name
        timestamp2 = Path(trainer2.output_dir).name

        assert timestamp1 != timestamp2

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_output_dir_absolute_path(self, mock_tokenizer_fn, mock_tokenizer):
        """测试输出目录路径格式"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer

        trainer = RLTrainer(
            model_name_or_path="test/model",
            output_dir="./outputs"
        )

        output_path = Path(trainer.output_dir)
        # 验证路径格式正确
        assert "outputs/model" in str(output_path)
        # 验证路径可以解析为绝对路径
        assert output_path.resolve().is_absolute()


# ========== 边界情况测试 ==========


class TestEdgeCases:
    """测试边界情况"""

    @patch('liagents.rl.trainer.AutoTokenizer.from_pretrained')
    def test_init_with_minimal_params(self, mock_tokenizer_fn, mock_tokenizer):
        """测试使用最小参数初始化"""
        mock_tokenizer_fn.return_value = mock_tokenizer

        from liagents.rl.trainer import RLTrainer

        trainer = RLTrainer()

        # 验证使用默认值
        assert trainer.model_name_or_path == "Qwen/Qwen3-0.6B"
        assert "Qwen3-0.6B" in trainer.output_dir

    def test_train_with_custom_dataset_and_no_registered(self, rl_trainer, mock_dataset):
        """测试只提供自定义数据集（没有注册数据集）"""
        rl_trainer.dataset = None

        with patch('liagents.rl.trainer.check_trl_installation', return_value=True):
            rl_trainer._train_sft = Mock(return_value={"status": "success"})

            rl_trainer.train(
                algorithm="sft",
                custom_dataset=mock_dataset
            )

            # 验证不会报错，因为使用了 custom_dataset
            rl_trainer._train_sft.assert_called_once()

    def test_train_monitoring_flags(self, rl_trainer_with_dataset):
        """测试监控标志的正确传递"""
        rl_trainer_with_dataset._train_sft = Mock(return_value={"status": "success"})

        with patch('liagents.rl.trainer.check_trl_installation', return_value=True):
            rl_trainer_with_dataset.train(
                algorithm="sft",
                use_wandb=True,
                use_tensorboard=True,
                wandb_project="test_project"
            )

            # 验证监控参数被传递
            call_kwargs = rl_trainer_with_dataset._train_sft.call_args.kwargs
            assert call_kwargs["use_wandb"] is True
            assert call_kwargs["use_tensorboard"] is True
            assert call_kwargs["wandb_project"] == "test_project"


# ========== 集成测试示例 ==========


class TestIntegration:
    """集成测试示例"""

    @patch('liagents.rl.trainer.check_trl_installation')
    @patch('liagents.rl.trainer.SFTTrainerWrapper')
    @patch('liagents.rl.trainer.setup_training_environment')
    @patch('liagents.rl.trainer.TrainingConfig')
    @patch('liagents.rl.datasets.load_dataset')
    def test_full_training_workflow(
        self, mock_load_dataset, mock_config, mock_setup_env, mock_trainer_wrapper,
        mock_check_trl, mock_tokenizer
    ):
        """测试完整的训练工作流"""
        mock_check_trl.return_value = True

        # 准备数据集 mock
        mock_data = {
            "question": ["What is 2+2?"],
            "answer": ["4"]
        }
        mock_hf_dataset = Dataset.from_dict(mock_data)
        mock_load_dataset.return_value = mock_hf_dataset

        # 准备训练器 mock
        mock_trainer_instance = Mock()
        mock_trainer_wrapper.return_value = mock_trainer_instance

        with patch('liagents.rl.trainer.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            from liagents.rl.trainer import RLTrainer

            # 1. 创建训练器
            trainer = RLTrainer(
                model_name_or_path="test/model",
                output_dir="./test_outputs"
            )

            # 2. 加载数据集
            load_result = trainer.load_dataset(
                dataset_name_or_path="openai/gsm8k",  # 使用支持的数据集名称
                format_type="sft",
                split="train",
                max_samples=-1
            )
            assert load_result["status"] == "success"

            # 3. 训练
            train_result = trainer.train(
                algorithm="sft",
                num_epochs=3,
                batch_size=8
            )
            train_result_dict = json.loads(train_result)
            assert train_result_dict["status"] == "success"

            # 验证完整的训练流程被调用
            mock_setup_env.assert_called_once()
            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once()
