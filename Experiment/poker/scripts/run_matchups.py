from poker.envs import PokerEnv
from poker.agents import LLMAgent, PokerActionParser
from poker.agents.model_runners import LlamaLoRAModelRunner
from poker.coordinator.coordinator import PokerCoordinator, run_evaluation


def load_agents():
    base_model = "meta-llama/Meta-Llama-3-8B"
    parser = PokerActionParser()

    # Base
    llama_runner = LlamaLoRAModelRunner(base_model, adapter_dir=None)
    agent_base = LLMAgent("Llama3-8B-Base", llama_runner, parser)

    # LoRA SFT
    sft_runner = LlamaLoRAModelRunner(base_model, adapter_dir="lora_poker_model")
    agent_sft = LLMAgent("LoRA-Poker", sft_runner, parser)

    # PPO 1000 episodes
    ppo_runner = LlamaLoRAModelRunner(base_model, adapter_dir="PPO_1000_model")
    agent_ppo = LLMAgent("PPO-Poker", ppo_runner, parser)

    return agent_base, agent_sft, agent_ppo


def run_matchup(agentA, agentB, file_name, hands=100):
    print(f"\n=== Running {agentA.name} vs {agentB.name} ===")

    env = PokerEnv(num_players=2, starting_stack=100.0)

    coordinator = PokerCoordinator(
        env,
        agents={0: agentA, 1: agentB},
        history_path=file_name
    )

    wins, chip_diff, bb_100 = run_evaluation(env, coordinator, num_hands=hands)

    print("\n===== RESULTS =====")
    for pid, agent in coordinator.agents.items():
        print(f"{agent.name}:")
        print(f"  Wins       : {wins[pid]}")
        print(f"  Chips Won  : {chip_diff[pid]:.2f}")
        print(f"  BB/100     : {bb_100[pid]:.2f}")
        print()


def main():
    agent_base, agent_sft, agent_ppo = load_agents()

    # 1. SFT vs PPO
    run_matchup(agent_sft, agent_ppo, "hand_history_SFT_vs_PPO.jsonl")

    # 2. Base vs PPO
    run_matchup(agent_base, agent_ppo, "hand_history_Base_vs_PPO.jsonl")

    # 3. Base vs SFT
    run_matchup(agent_base, agent_sft, "hand_history_Base_vs_SFT.jsonl")


if __name__ == "__main__":
    main()

