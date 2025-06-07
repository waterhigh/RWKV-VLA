# builder.py

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
import argparse # 用于为测试创建配置对象

# ====================================================================================
# 关键依赖项导入
# 确保这些文件相对于您的项目根目录是可访问的
# 如果运行时出现 ModuleNotFoundError，您可能需要调整这些导入路径
# 例如，去掉前面的 '.' (from .hidden_model -> from hidden_model)
# ====================================================================================
from src.hidden_model import VisualRWKV
from policy_heads.models.droid_unet_diffusion import ConditionalUnet1D, SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import process_image_tokens_in_conversations, preprocess
from src.utils import Conversation

# 定义一些VLM可能需要的常量
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

class RWKVDroidPolicy(nn.Module):
    """
    集成的策略模型，内部包含了完整的VLM输入预处理逻辑，并使用真实组件。
    """
    def __init__(self,
                 # 文件路径
                 vlm_model_path: str,
                 tokenizer_path: str,
                 # VLM 配置
                 vlm_config_args,
                 # Diffusion Head & Action 配置
                 action_dim: int,
                 state_dim: int,
                 action_horizon: int,
                 down_dims: list = [256, 512, 1024],
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 # Scheduler 配置
                 num_train_timesteps: int = 100,
                 num_inference_steps: int = 10,
                 beta_schedule: str = 'squaredcos_cap_v2',
                 prediction_type: str = 'epsilon'
                ):
        super().__init__()
        
        # 保存动作/状态配置
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        
        # 1. 初始化 VLM 和 Tokenizer (使用真实组件和路径)
        print("正在初始化 VisualRWKV...")
        self.vlm = VisualRWKV(vlm_config_args)
        print(f"正在从 {vlm_model_path} 加载VLM权重...")
        self.vlm.load_state_dict(torch.load(vlm_model_path, map_location='cpu'), strict=False)
        self.vlm.eval().bfloat16() # 设置为评估模式和bf16
        
        print(f"正在从 {tokenizer_path} 初始化 TRIE_TOKENIZER...")
        self.tokenizer = TRIE_TOKENIZER(tokenizer_path)
        
        # 2. 冻结VLM
        print("正在冻结 VisualRWKV 的参数...")
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        self.global_cond_dim = vlm_config_args.n_embd
        print(f"VLM 全局条件维度已确认为: {self.global_cond_dim}")

        # 3. 初始化 Diffusion Head
        print("正在初始化 ConditionalUnet1D Diffusion Head...")
        self.diffusion_head = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.global_cond_dim,
            state_dim=state_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )
        
        # 4. 初始化噪声调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=prediction_type
        )
        
        # 保存预处理需要的参数
        self.ctx_len = vlm_config_args.ctx_len
        self.image_position = getattr(vlm_config_args, 'image_position', 'first')

    def _prepare_vlm_input(self, image_input, text_instructions, device):
        """一个私有方法，封装了从demo.py学到的完整预处理逻辑。"""
        batch_size = image_input.shape[0]
        all_input_ids = []
        all_labels = []

        for i in range(batch_size):
            instruction = text_instructions[i]
            input_text = DEFAULT_IMAGE_TOKEN + ' ' + instruction

            conv = Conversation(id=f"session_{i}", roles=["human", "gpt"], conversations=[])
            conv.append_message(conv.roles[0], input_text)
            
            conversations_processed = process_image_tokens_in_conversations(
                conv.conversations, image_position=self.image_position
            )
            
            data_dict = preprocess(
                conversations_processed,
                self.tokenizer,
                has_image=True,
                ctx_len=self.ctx_len,
                pad_token_id=0,
                do_pad_to_max_length=False
            )
            all_input_ids.append(data_dict['input_ids'])
            all_labels.append(data_dict['labels'])
        
        max_len = max(len(ids) for ids in all_input_ids)
        
        padded_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            len_ids = len(all_input_ids[i])
            padded_input_ids[i, :len_ids] = all_input_ids[i]
            padded_labels[i, :len_ids] = all_labels[i]

        samples = {
            "images": image_input.bfloat16(),
            "input_ids": padded_input_ids,
            "labels": padded_labels
        }
        return samples

    def forward(self, image_input, text_instructions, robot_state, ground_truth_actions):
        """训练时的前向传播"""
        device = image_input.device
        
        with torch.no_grad():
             samples = self._prepare_vlm_input(image_input, text_instructions, device)
             global_condition = self.vlm.get_global_condition(samples)
        
        noise = torch.randn_like(ground_truth_actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (ground_truth_actions.shape[0],), device=device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)
        
        predicted_noise = self.diffusion_head(
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=global_condition,
            states=robot_state
        )
        
        return predicted_noise, noise

# ====================================================================================
#                           主测试用例 (使用真实组件)
# ====================================================================================

if __name__ == '__main__':
    print("="*60)
    print("RUNNING RWKVDroidPolicy REAL INTEGRATION TEST")
    print("="*60)

    # ##################################################################
    # ### 用户：请在此处修改为您的实际文件路径                       ###
    # ##################################################################
    VLM_MODEL_PATH = "/home/bgi/code/VLA/weights/VisualRWKV-v060-1B6-v1.0-20240612.pth"
    TOKENIZER_PATH = "src/rwkv_vocab_v20230424.txt"
    # ##################################################################

    # --- 1. 设置测试参数 ---
    ACTION_DIM = 7  # 例如: 6-DoF末端执行器 + 1夹爪状态
    STATE_DIM = 7   # 例如: 机器人自身的本体感受状态
    ACTION_HORIZON = 16
    BATCH_SIZE = 2  # 使用一个较小的批次大小以避免内存问题
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Test Configuration:")
    print(f"  - Device: {DEVICE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Action Dim: {ACTION_DIM}")
    print(f"  - State Dim: {STATE_DIM}")
    print(f"  - VLM Model: {VLM_MODEL_PATH}")
    print(f"  - Tokenizer: {TOKENIZER_PATH}")

    # --- 2. 创建VLM所需的配置对象 (模仿您的demo.py) ---
    vlm_config_args = argparse.Namespace(
        n_layer=24,
        n_embd=2048,
        ctx_len=256, # 可以根据需要调整
        vocab_size=65536,
        dim_att=2048,
        dim_ffn=7168,
        pre_ffn=0,
        head_size_a=64,
        head_size_divisor=8,
        dropout=0.0,
        grad_cp=0,
        grid_size=-1,
        image_position='first'
    )
    os.environ["RWKV_HEAD_SIZE_A"] = str(vlm_config_args.head_size_a)

    # --- 3. 实例化我们的主策略模型 ---
    try:
        policy = RWKVDroidPolicy(
            vlm_model_path=VLM_MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            vlm_config_args=vlm_config_args,
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            action_horizon=ACTION_HORIZON
        ).to(DEVICE)
    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        print("请确保上面的 VLM_MODEL_PATH 和 TOKENIZER_PATH 是正确的。")
        exit()
    except Exception as e:
        print(f"[ERROR] 初始化 RWKVDroidPolicy 时出错: {e}")
        import traceback
        traceback.print_exc()
        exit()
        
    print("--- RWKVDroidPolicy Initialized Successfully ---")

    # --- 4. 创建随机输入数据 ---
    dummy_image_input = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
    dummy_text_instructions = [f"pick up the red block" for _ in range(BATCH_SIZE)]
    dummy_robot_state = torch.randn(BATCH_SIZE, STATE_DIM, device=DEVICE, dtype=torch.bfloat16)
    dummy_ground_truth_actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM, device=DEVICE, dtype=torch.bfloat16)

    print("--- Testing Training Forward Pass ---")
    print(f"Input action shape: {dummy_ground_truth_actions.shape}")
    
    # --- 5. 运行一次前向传播 ---
    try:
        # 将模型设置为训练模式 (尽管VLM是冻结的，但diffusion head需要)
        policy.train()
        
        predicted_noise, original_noise = policy.forward(
            image_input=dummy_image_input,
            text_instructions=dummy_text_instructions,
            robot_state=dummy_robot_state,
            ground_truth_actions=dummy_ground_truth_actions
        )
        
        print("--- FORWARD PASS COMPLETED ---")
        print(f"Output predicted_noise shape: {predicted_noise.shape}")
        
        # --- 6. 验证输出形状 ---
        assert predicted_noise.shape == dummy_ground_truth_actions.shape
        print("[SUCCESS] Output shape matches input action shape.")
        
        print("" + "="*60)
        print("🎉 INTEGRATION TEST PASSED! The full forward pass works correctly. 🎉")
        print("="*60)

    except Exception as e:
        print("" + "="*60)
        print("❌ TEST FAILED! An error occurred during the forward pass:")
        import traceback
        traceback.print_exc()
        print("="*60)
