# builder.py

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
os.environ["RWKV_JIT_ON"] = "1" 
os.environ["RWKV_HEAD_SIZE_A"] = str(64)
os.environ["RWKV_CTXLEN"] = str(256) 
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
        self.state_dim = state_dim # 保存状态维度，推理时需要
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
        
        self.diffusion_head.to(dtype=torch.bfloat16)

        # 4. 初始化噪声调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=True, # 建议开启，防止动作值超出范围
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

    @torch.no_grad()
    def plan_action(self, image_input, text_instruction, robot_state):
        """
        推理/规划函数，用于在实际环境中生成动作序列。
        
        Args:
            image_input (torch.Tensor): 单帧图像, shape: [1, 1, C, H, W]
            text_instruction (str): 单个文本指令, e.g. "pick up the red block"
            robot_state (torch.Tensor): 当前机器人状态, shape: [1, state_dim]
            
        Returns:
            torch.Tensor: 规划出的动作序列, shape: [1, action_horizon, action_dim]
        """
        # 确保模型处于评估模式
        self.eval()
        device = image_input.device
        dtype = next(self.diffusion_head.parameters()).dtype # 获取diffusion head的数据类型 (bfloat16)

        # 1. 使用VLM提取全局条件
        # 注意：这里我们把单个指令包装成一个列表，以匹配_prepare_vlm_input的输入要求
        samples = self._prepare_vlm_input(image_input, [text_instruction], device)
        global_condition = self.vlm.get_global_condition(samples)
        
        # 2. 准备去噪过程的初始输入
        # 初始化一个纯噪声张量作为动作序列的起点
        noisy_actions = torch.randn(
            (1, self.action_horizon, self.action_dim), 
            device=device, 
            dtype=dtype
        )
        # 将机器人状态转换为正确的类型和设备
        robot_state = robot_state.to(device=device, dtype=dtype)

        # 3. 设置调度器的时间步
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # 4. 逐步去噪循环
        for t in self.noise_scheduler.timesteps:
            # 预测当前时间步的噪声
            # 注意：timestep也需要是一个tensor
            timestep_tensor = t.unsqueeze(0).to(device)
            
            predicted_noise = self.diffusion_head(
                sample=noisy_actions,
                timestep=timestep_tensor,
                global_cond=global_condition,
                states=robot_state
            )
            
            # 使用调度器的step方法，从当前带噪动作中移除预测的噪声
            # 得到一个更清晰的动作序列
            noisy_actions = self.noise_scheduler.step(
                model_output=predicted_noise,
                timestep=t,
                sample=noisy_actions
            ).prev_sample

        # 5. 返回最终去噪后的动作序列
        return noisy_actions

# ====================================================================================
#                           主测试用例 (使用真实组件)
# ===================================================================================
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
    ACTION_DIM = 7
    STATE_DIM = 7
    ACTION_HORIZON = 16
    BATCH_SIZE = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Test Configuration:")
    print(f"  - Device: {DEVICE}")
    print(f"  - Batch Size (for training test): {BATCH_SIZE}")
    print(f"  - Action Dim: {ACTION_DIM}")
    print(f"  - State Dim: {STATE_DIM}")
    print(f"  - VLM Model: {VLM_MODEL_PATH}")
    print(f"  - Tokenizer: {TOKENIZER_PATH}")

    # --- 2. 创建VLM所需的配置对象 ---
    vlm_config_args = argparse.Namespace(
        n_layer=24, n_embd=2048, ctx_len=256, vocab_size=65536,
        load_model="", vision_tower_name="/home/bgi/code/VLA/weights/CLIP",
        dim_att=2048, dim_ffn=7168, pre_ffn=0, head_size_a=64,
        head_size_divisor=8, dropout=0.0, grad_cp=0, grid_size=-1,
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
    except Exception as e:
        print(f"[ERROR] 初始化 RWKVDroidPolicy 时出错: {e}")
        import traceback
        traceback.print_exc()
        exit()
        
    print("--- RWKVDroidPolicy Initialized Successfully ---")

    # ======================== 训练前向传播测试 ========================
    print("--- Testing Training Forward Pass ---")
    
    # 创建训练用的批处理数据
    dummy_train_images = torch.randn(BATCH_SIZE, 1, 3, 336, 336, device=DEVICE)
    dummy_train_instructions = [f"pick up object {i}" for i in range(BATCH_SIZE)]
    dummy_train_state = torch.randn(BATCH_SIZE, STATE_DIM, device=DEVICE, dtype=torch.bfloat16)
    dummy_train_actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM, device=DEVICE, dtype=torch.bfloat16)
    
    try:
        policy.train()
        predicted_noise, original_noise = policy.forward(
            image_input=dummy_train_images,
            text_instructions=dummy_train_instructions,
            robot_state=dummy_train_state,
            ground_truth_actions=dummy_train_actions
        )
        print("--- TRAINING FORWARD PASS COMPLETED ---")
        print(f"Output predicted_noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == dummy_train_actions.shape
        print("[SUCCESS] Output shape matches ground truth action shape.")
        
    except Exception as e:
        print("" + "="*60)
        print("❌ TRAINING TEST FAILED! An error occurred:")
        import traceback
        traceback.print_exc()
        exit()

    print("" + "="*60 + "")

    # ======================== 推理/规划函数测试 ========================
    print("--- Testing Inference/Planning Function (plan_action) ---")
    
    # 创建推理用的单样本数据
    dummy_inference_image = torch.randn(1, 1, 3, 336, 336, device=DEVICE)
    dummy_inference_instruction = "put the green cup into the sink"
    dummy_inference_state = torch.randn(1, STATE_DIM, device=DEVICE)

    try:
        planned_actions = policy.plan_action(
            image_input=dummy_inference_image,
            text_instruction=dummy_inference_instruction,
            robot_state=dummy_inference_state
        )
        print("--- INFERENCE (plan_action) COMPLETED ---")
        print(f"Output planned_actions shape: {planned_actions.shape}")
        
        # 验证输出形状
        expected_shape = (1, ACTION_HORIZON, ACTION_DIM)
        assert planned_actions.shape == expected_shape
        print(f"[SUCCESS] Output shape {planned_actions.shape} matches expected shape {expected_shape}.")

        print("" + "="*60)
        print("🎉 ALL TESTS PASSED! The model works for both training and inference. 🎉")
        print("="*60)

    except Exception as e:
        print("" + "="*60)
        print("❌ INFERENCE TEST FAILED! An error occurred in plan_action:")
        import traceback
        traceback.print_exc()
        exit()
