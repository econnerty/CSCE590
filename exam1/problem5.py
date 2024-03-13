# Define the parameters given in the problem
essay_time = 10  # minutes per essay question
short_time = 2   # minutes per short-answer question
total_time = 90  # total time for the exam
min_essays = 3   # minimum number of essay questions required
essay_points = 20  # points per essay question
short_points = 5   # points per short-answer question
max_essays = 10   # total available essay questions
max_shorts = 50   # total available short-answer questions

# Initialize maximum score and corresponding number of questions
max_score = 0
best_essay = 0
best_short = 0

# Iterate through the possible number of essay questions
for essays in range(min_essays, max_essays + 1):
    # Calculate remaining time after doing the minimum essay questions
    remaining_time = total_time - essays * essay_time
    
    # Calculate the maximum number of short-answer questions that can be done in the remaining time
    shorts = min(remaining_time // short_time, max_shorts)
    
    # Calculate the score for this combination
    score = essays * essay_points + shorts * short_points
    
    # If this score is better than the current best, update max_score and remember the number of questions
    if score > max_score:
        max_score = score
        best_essay = essays
        best_short = shorts

print(max_score, best_essay, best_short)

