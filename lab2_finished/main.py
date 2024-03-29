from genetic import * 


if __name__ == "__main__":


    best_agent = evolution_strategy()
    np.save("genome_strategy.npy", best_agent.genome_strategy)
    np.save("genome_row.npy", best_agent.genome_row)

    nim = Nim(NUM_ROWS)

    total_win = 0
    print("Random vs Adaptive")
    for i in range(10):
        print(f"First Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(adaptive, pure_random)
        print(f"Winner: Player{winner}")
        if winner ==0:
            total_win += 1
    for i in range (10):
        print(f"Second Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(pure_random, adaptive)
        print(f"Winner: Player{winner}")
        if winner ==1:
            total_win += 1

    print(f"Total win: {total_win}")



    print("Gabriele vs Adaptive")
    total_win = 0
    for i in range(10):
        print(f"First Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(adaptive, gabriele)
        print(f"Winner: Player{winner}")
        if winner ==0:
            total_win += 1
    for i in range (10):
        print(f"Second Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(gabriele, adaptive)
        print(f"Winner: Player{winner}")
        if winner ==1:
            total_win += 1

    print(f"Total win: {total_win}")


    print("Optimal vs Adaptive")

    total_win = 0
    for i in range(10):
        print(f"First Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(adaptive, optimal)
        print(f"Winner: Player{winner}")
        if winner ==0:
            total_win += 1
    for i in range (10):
        print(f"Second Moves Games {i+1}")
        nim.reset()
        winner , _, _=  nim.play(optimal, adaptive)
        print(f"Winner: Player{winner}")
        if winner ==1:
            total_win += 1

    print(f"Total win: {total_win}")









        
