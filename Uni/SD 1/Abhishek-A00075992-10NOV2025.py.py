name = (input("Enter your name: "))
cw1 = float(input("Enter your CW1 marks: "))
cw2 = float(input("Enter your CW2 marks: "))
cw3 = float(input("Enter your CW3 marks: "))
final_exam_marks = float(input("Enter your final exam marks: "))

complete_rank = (cw1 * 0.10) + (cw2 * 0.20) + (cw3 * 0.30) + (final_exam_marks * 0.40)
complete_rank = round(complete_rank)

if complete_rank == 100:
    result = "- Aurum Standard"
elif 92 <= complete_rank <= 99:
    result = "- Upper First"
elif 82 <= complete_rank < 92:
    result = "- Upper First"
elif 78 <= complete_rank < 82:
    result = "- First"
elif 72 <= complete_rank < 78:
    result = "- First"
elif 68 <= complete_rank < 72:
    result = "- 2:1 Upper"
elif 62 <= complete_rank < 68:
    result = "- 2:1 Upper"
elif 58 <= complete_rank < 62:
    result = "- 2:2 Lower"
elif 52 <= complete_rank < 58:
    result = "- 2:2 Lower"
elif 48 <= complete_rank < 52:
    result = "- Third"
elif 42 <= complete_rank < 48:
    result = "- Third"
elif 32 <= complete_rank < 42:
    result = "- Condonable Fail"
elif 5 <= complete_rank < 32:
    result = "- Fail"
elif complete_rank == 0:
    result = "- Deficit Opus"
else:
    result = "Invalid Input"
print(name, complete_rank, result)