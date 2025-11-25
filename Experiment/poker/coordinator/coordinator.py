import json

class PokerCoordinator:
    """
    Coordinates agent actions and stores full hand history.
    """

    def __init__(self, env, agents, history_path="hand_history.jsonl"):
        self.env = env
        self.agents = agents
        self.history_path = history_path

    def play_hand_with_history(self):
        state = self.env.reset()
        done = False
        info = None

        hand_hist = {
            "hand_number": self.env.game.state.hand_number,
            "players": {},
            "actions": [],
            "streets": {},
            "result": None
        }

        # initial hole cards
        for p in self.env.game.players:
            hand_hist["players"][p.player_id] = {
                "position": p.position,
                "starting_stack": p.stack,
                "hole_cards": [str(c) for c in p.hole_cards],
            }

        # play hand
        while not done:
            p = state.current_player()
            pid = p.player_id

            obs = self.env.get_observation(pid)
            valid = self.env.get_valid_actions_for_player(pid)

            action_name, amt = self.agents[pid].act(obs, valid)
            act_enum = self.env.string_to_action_enum(action_name)

            hand_hist["actions"].append({
                "player_id": pid,
                "action": action_name,
                "amount": amt,
                "betting_round": state.betting_round.value,
                "stack_after_action": p.stack,
            })

            state, done, info = self.env.step(act_enum, amt)

            br = state.betting_round.value
            hand_hist["streets"][br] = [
                str(c) for c in self.env.game.state.community_cards
            ]

        hand_hist["result"] = info
        return hand_hist

    def save_hand_history(self, hand_hist):
        with open(self.history_path, "a") as f:
            f.write(json.dumps(hand_hist) + "\n")


def run_evaluation(env, coordinator, num_hands=100):
    wins = {pid: 0 for pid in coordinator.agents.keys()}
    chips_won = {pid: 0.0 for pid in coordinator.agents.keys()}

    prev_stacks = {pid: env.game.players[pid].stack
                   for pid in coordinator.agents.keys()}

    bb = env.game.big_blind

    for hand in range(1, num_hands + 1):

        # stop early if someone busts
        if any(player.stack <= 0 for player in env.game.players):
            print(f"[STOP] Player busted on hand {hand-1}. Ending early.")
            break

        hand_hist = coordinator.play_hand_with_history()
        coordinator.save_hand_history(hand_hist)

        info = hand_hist["result"]
        for wid in info["winners"]:
            wins[wid] += 1

        # update chip diff
        for pid, player in enumerate(env.game.players):
            final_stack = player.stack
            chips_won[pid] += final_stack - prev_stacks[pid]
            prev_stacks[pid] = final_stack

        if hand % 10 == 0:
            print(f"[Progress] {hand}/{num_hands} hands")

    total_hands = hand - 1

    # compute bb/100
    bb_100 = {
        pid: (chips_won[pid] / bb) / (total_hands / 100)
        for pid in chips_won
    }

    return wins, chips_won, bb_100
