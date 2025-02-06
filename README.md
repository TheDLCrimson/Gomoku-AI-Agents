# **Gomoku AI Agents**  

## **Overview**  
This project focuses on developing **AI agents** for Gomoku, a strategic board game similar to Tic Tac Toe. The goal is to implement and evaluate agents using various AI algorithms, including **Reflex, Minimax with Alpha-Beta Pruning, Q-Learning, and Monte Carlo Tree Search (MCTS)**. The project demonstrates the effectiveness of each strategy and explores advanced techniques to enhance agent performance.  

---

## **Key Objectives**  
1. Implement **4 AI agents** using diverse algorithms: Reflex, Minimax, Q-Learning, and MCTS.  
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

4. **Better MCTS Agent:**  
   - Enhances MCTS with **heuristic rollouts**, **progressive widening**, and **memoization**.  
   - Scored **47/70** in evaluations with 1000 simulations.  

---

## **Methodology**  
### **1. Reflex Agent**  
- Evaluates potential moves using a **weighted scoring system** based on stone patterns.  
- Prioritizes moves that create or block sequences of stones.  

### **2. Minimax with Alpha-Beta Pruning**  
- Uses **recursive search** to evaluate player and opponent moves.  
- Optimizes search with **Alpha-Beta pruning** to reduce computation time.  

### **3. Approximate Q-Learning Agent**  
- Extracts features (e.g., open threes, fours) using **SimpleExtractor**.  
- Trains with **Q-learning** and an **epsilon-greedy strategy** for decision-making.  

### **4. Better MCTS Agent**  
- Combines **MCTS** with **Alpha-Beta search** for deeper exploration.  
- Implements **heuristic rollouts** and **progressive widening** to improve efficiency.  

---

## **Results**  
- **Reflex Agent:** Effective against beginner bots but struggles with advanced strategies.  
- **Minimax Agent:** Outperformed intermediate bots with **9 wins, 0 losses, 1 draw**.  
- **Q-Learning Agent:** Achieved an evaluation score of **65/70**, showing strong learning capabilities.  
- **Better MCTS Agent:** Scored **47/70** with 1000 simulations, demonstrating improved strategic depth.  

---

## **Key Insights**  
1. **Heuristic-Based Rollouts:** Improved MCTS performance by guiding simulations with domain knowledge.  
2. **Hybrid Search:** Combining **MCTS** and **Alpha-Beta** enhanced strategic depth and efficiency.  
3. **Q-Learning:** Showed potential for adaptive strategies but required extensive training.  

---

## **Conclusion**  
This project highlights the effectiveness of **heuristic-based rollouts** and **hybrid search techniques** in improving AI performance for Gomoku. Future work could explore **deep reinforcement learning** (e.g., AlphaGo Zero) and **larger board sizes** to further enhance agent adaptability and strategic depth.  

---
