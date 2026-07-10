"""
Student Grading System
A complete grading program that collects student information, calculates overall scores,
determines categories with custom rounding rules, calculates ages, and displays/saves results.

Features:
- Configurable module setup (partial - uses fixed components as per original CW1)
- Input validation for ID, DOB, and scores
- Age calculation using datetime
- Raw and adjusted (rounded) score calculation
- Tabulated output sorted by student ID
- Results saved to 'students.txt'
"""

from datetime import datetime, date
from tabulate import tabulate


def calculate_overall_score():
    """
    Calculate the weighted overall score using fixed weights:
    Coursework 1: 10%, Coursework 2: 20%, Coursework 3: 30%, Final Exam: 40%
    """
    overall_score = c1 * 0.10 + c2 * 0.20 + c3 * 0.30 + fe * 0.40
    return overall_score


def determine_category(overallscore):
    """
    Adjust the raw score to the nearest 'representative' category score
    based on the provided banding rules, then determine the classification category.
    
    Returns: (adjusted_score, category_name)
    """
    adjusted = overallscore

    # Custom adjustment rules to snap scores into category bands
    if 93 <= adjusted <= 100:
        adjusted = 100
    elif 79 <= adjusted <= 81:
        adjusted = 82
    elif 69 <= adjusted <= 71:
        adjusted = 72
    elif 61 <= adjusted <= 67:
        adjusted = 68
    elif 59 <= adjusted <= 61:
        adjusted = 62
    elif 49 <= adjusted <= 51:
        adjusted = 52
    elif 39 <= adjusted <= 41:
        adjusted = 42
    elif 21 <= adjusted <= 31:
        adjusted = 25

    # Determine classification category based on adjusted score
    if adjusted == 100:
        category = "- Aurum Standard"
    elif 82 <= adjusted <= 92:
        category = "- Upper First"
    elif 72 <= adjusted <= 78:
        category = "- First"
    elif 62 <= adjusted <= 68:
        category = "- 2:1"
    elif 52 <= adjusted <= 58:
        category = "- 2:2"
    elif 42 <= adjusted <= 48:
        category = "- Third"
    elif 32 <= adjusted <= 38:
        category = "- Condonable Fail"
    elif 5 <= adjusted <= 25:
        category = "- Fail"
    elif adjusted == 0:
        category = "- Defecit Opus"
    else:
        category = "- Invalid overallscore"

    return adjusted, category


def main():
    """
    Main function: Orchestrates the entire program flow
    - Welcomes user
    - Collects up to 3 students (or until 'end')
    - Validates all inputs
    - Calculates scores and ages
    - Displays and saves results in a formatted table
    """
    print("Welcome to the Student Grading System")
    print("First, let's set up the module configuration.\n")

    # Simple module configuration prompt (using fixed components as per original design)
    configure = input("Configure new module? (yes/no): ").strip().lower()
    if configure == "yes":
        module_name = input("Enter module name: ")
        print(f"Module '{module_name}' selected (using default 4 components with fixed weights).")
    else:
        print("Using default module configuration (Coursework 1-3 + Final Exam).")

    print("\nNow, let's enter student details and their scores.\n")

    students = []
    global c1, c2, c3, fe  # Needed because calculate_overall_score uses these variables

    while len(students) < 3:
        print(f"\n--- Student {len(students) + 1} ---")

        # --- Student ID Input & Validation ---
        student_id = input("Enter student ID (2 digits) or 'end' to finish: ").strip()
        if student_id.lower() == "end":
            break
        if not student_id.isdigit() or len(student_id) != 2:
            print("The input you entered was invalid (ID must be exactly 2 digits)")
            continue

        # --- Name Input ---
        name = input("Enter student name: ").strip()

        # --- Date of Birth Input & Validation ---
        valid_dob = False
        while not valid_dob:
            try:
                dob_str = input("Enter date of birth (yyyy-mm-dd): ").strip()
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                valid_dob = True
            except ValueError:
                print("The input you entered was invalid (use yyyy-mm-dd format)")

        # --- Calculate Age ---
        today = date.today()
        age = today.year - dob.year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1

        # --- Score Inputs with Validation ---
        print("Enter component scores (0-100):")
        valid_scores = False
        while not valid_scores:
            try:
                c1 = int(input("Enter Coursework 1 score: "))
                c2 = int(input("Enter Coursework 2 score: "))
                c3 = int(input("Enter Coursework 3 score: "))
                fe = int(input("Enter Final Exam score: "))

                if all(0 <= score <= 100 for score in [c1, c2, c3, fe]):
                    valid_scores = True
                else:
                    print("The input you entered was invalid (all scores must be between 0 and 100)")
            except ValueError:
                print("The input you entered was invalid (please enter integers only)")

        # --- Calculate Raw and Adjusted Scores ---
        raw_score = calculate_overall_score()
        adjusted_score, category = determine_category(raw_score)

        # --- Store Student Record ---
        students.append({
            "UID": student_id,
            "Name": name,
            "D.o.B": dob.strftime("%Y-%m-%d"),
            "Age": age,
            "Raw Score": round(raw_score, 4),
            "Rounded Score": adjusted_score,
            "Category": category
        })

    # --- Final Processing if Students Were Entered ---
    if students:
        # Sort students by UID (ascending order)
        students.sort(key=lambda x: x["UID"])

        # Define table headers
        headers = ["UID", "Name", "D.o.B", "Age", "Raw Score", "Rounded Score", "Category"]

        # Display results in a nice table
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        table_output = tabulate(students, headers=headers, tablefmt="grid")
        print(table_output)

        # Save results to file
        with open("students.txt", "w") as file:
            file.write(table_output)
        print("\nResults have been saved to 'students.txt'")
    else:
        print("\nNo student data was entered. Exiting.")


# --- Program Entry Point ---
if __name__ == "__main__":
    main()