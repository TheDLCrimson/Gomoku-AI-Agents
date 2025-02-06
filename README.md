# **Gomoku AI Agents**

## **Overview**

This project focuses on developing **AI agents** for Gomoku, a strategic board game similar to Tic Tac Toe. The goal is to implement and evaluate agents using various AI algorithms, including **Reflex, Minimax with Alpha-Beta Pruning, Q-Learning, and Monte Carlo Tree Search (MCTS)**. The project demonstrates the effectiveness of each strategy and explores advanced techniques to enhance agent performance.

---

## **Key Objectives**

1. Implement **4 main AI agents** using diverse algorithms: Reflex, Minimax, Q-Learning, and MCTS.
2. Evaluate agent performance against pre-built bots and human players.
3. Explore advanced techniques like **heuristic-based rollouts** and **hybrid MCTS-AlphaBeta search** to improve strategic depth and efficiency.

---

## **Implemented Agents**

1. **Reflex Agent:**

- Uses **heuristic evaluation** to prioritize moves based on stone patterns.
- Achieved **40 wins, 10 losses** against beginner bots.

2. **Minimax with Alpha-Beta Pruning:**

- Implements **adversarial search** with pruning to optimize decision-making.
- Scored **9 wins, 0 losses, 1 draw** against intermediate bots.

3. **Approximate Q-Learning Agent:**

- Uses **feature extraction** and **Q-learning** to learn optimal strategies.
- Achieved an evaluation score of **65/70** after 150 training episodes.

4. **MCTS Agent:**

- Naive implementation with **2000 simulations** in 8x8 board scores **3 wins, 0 losses, and 7 draws** & **40/70** in evaluations.
- Better MCTS with **heuristic rollouts**, **progressive widening**, and **memoization** with **1000 simulations** scores **47/70** in evaluations

---

## **Methodology**

### **1. Reflex Agent**

- Evaluates potential moves using a **weighted scoring system** based on stone patterns.
- Prioritizes moves that create or block sequences of stones.

### **2. Minimax with Alpha-Beta Pruning**

- Uses **recursive search** to evaluate player and opponent moves.
- Optimizes search with **Alpha-Beta pruning** to reduce computation time.
- A **heuristic evaluation** function scores board states based on patterns like open fours and blocked threes.
- Orders moves with **PromisingNextMoves**, prioritizing those with higher strategic value.

### **3. Approximate Q-Learning Agent**

- Extracts features (e.g., open threes, fours) using **SimpleExtractor**.
- Trains with **Q-learning** and an **epsilon-greedy strategy** for decision-making.
- Learns optimal move selection by **adjusting weights** based on gameplay outcomes.

### **4. Better MCTS Agent**

- Combines **MCTS** with **Alpha-Beta search** for deeper exploration.
- Implements **heuristic rollouts** and **progressive widening** to improve efficiency.
- Uses **memoization** and **improved node selection** to optimize simulations and search efficiency.

---

## **Results**

1. **Reflex Agent:**

   - This agent relies on immediate board evaluation without deep lookahead.
   - It performed well against beginner-level opponents, achieving **40 wins and 10 losses**.
   - Despite its reactive nature, it secured an **evaluation score of 50/70**, indicating moderate effectiveness.

2. **Minimax (Alpha-Beta Pruning) Agent:**

   - The Minimax algorithm, enhanced with Alpha-Beta pruning, showed superior strategic planning.
   - Against **Intermediate-level opponents**, it achieved an **impressive 9 wins, 0 losses, and 1 draw**, earning an **evaluation score of 66/70**.
   - Against **Master-level opponents**, it maintained strong performance with **8 wins and 2 losses**, demonstrating its advanced decision-making abilities.

3. **Monte Carlo Tree Search (MCTS) Agent:**

   - The **Naive MCTS agent**, using **2000 simulations**, performed against Intermediate-level opponents on an 8x8 board.
   - It managed **3 wins, 0 losses, and 7 draws**, leading to an **evaluation score of 40/70**, showing potential but limited effectiveness due to its simulation constraints.
   - The **Better MCTS agent**, refined with **1000 simulations**, improved upon the naive version, achieving an **evaluation score of 47/70**, indicating stronger strategic depth and adaptability.

4. **Approximate Q-Learning Agent:**
   - This reinforcement learning model was trained with **150 episodes** to refine decision-making through self-play and reward-based learning.
   - It achieved one of the highest scores, an **evaluation of 65/70**, suggesting effective learning and generalization.
   - The Q-Learning agent displayed strong adaptability and improvement over time, making it a competitive approach.

---

## **Key Insights**

1. **Heuristic-Based Rollouts:** Improved MCTS performance by guiding simulations with domain knowledge.
2. **Hybrid Search:** Combining **MCTS** and **Alpha-Beta** enhanced strategic depth and efficiency.
3. **Q-Learning:** Showed potential for adaptive strategies but required extensive training.
4. **Alpha-Beta Pruning + Move Ordering**: Significantly reduced search space, making Minimax more efficient without sacrificing decision quality.

---

## **Conclusion**

This project highlights the effectiveness of **heuristic-based rollouts** and **hybrid search techniques** in improving AI performance for Gomoku. Future work could explore **deep reinforcement learning** (e.g., AlphaGo Zero) and **larger board sizes** to further enhance agent adaptability and strategic depth.

---

## How to Run

Refer to [this README](https://github.com/TheDLCrimson/Gomoku-AI-Agents/blob/ae94bc08c070fb21db0bd7727efbfde4b8e0b7af/project/README.md) for installation and usage.
