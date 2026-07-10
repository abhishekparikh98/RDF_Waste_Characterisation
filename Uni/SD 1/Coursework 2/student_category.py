def determine_category(score):
    """
    REQUIREMENT 1 (part): determine_category()
    Uses the specified bands to convert a score to a category.
    You can adjust these bands to match your exact marking scheme.
    """
    # Adjust to special bands (your original style)
    if 93 <= score <= 99:
        score = 100
    elif 79 <= score <= 81:
        score = 82
    elif 69 <= score <= 71:
        score = 72
    elif 61 <= score <= 67:
        score = 68
    elif 59 <= score <= 61:
        score = 62
    elif 49 <= score <= 51:
        score = 52
    elif 39 <= score <= 41:
        score = 42
    elif 21 <= score <= 31:
        score = 25


    ranges = [
        (100, 100, "Aurum Standard"),
        (82, 92,  "Upper First"),
        (72, 78,  "First"),
        (62, 68,  "2:1"),
        (52, 58,  "2:2"),
        (42, 48,  "Third"),
        (32, 38,  "Condonable Fail"),
        (5,  25,  "Fail"),
        (0,   0,  "Defecit Opus"),
    ]

    category = "Invalid"  # default
    for low, high, name in ranges:
        if low <= score <= high:
            category = name
            break

    return score, category


# =====================================================
# REQUIREMENT 6: CALCULATE AGE
# =====================================================

def calculate_age(dob):
    """
    Takes a date object (DOB) and returns age in years.
    """
    today = date.today()
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


# =====================================================
# REQUIREMENT 9: INPUT VALIDATION HELPERS
# - ID, DOB, scores
# =====================================================

def input_student_id():
    """
    REQUIREMENT 9a: Validate student ID (2-digit or 'end').
    """
    while True:
        sid = input("Enter student ID (2 digit) or 'end' to stop: ")
        if sid.lower() == "end":
            return "end"
        if sid.isdigit() and len(sid) == 2:
            return sid
        print("The input you entered was invalid")


def input_dob():
    """
    REQUIREMENT 9a: Validate D.o.B. in YYYY-MM-DD format.
    """
    while True:
        dob_str = input("Enter date of birth (YYYY-MM-DD): ")
        try:
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            return dob
        except ValueError:
            print("The input you entered was invalid")


def input_score(prompt):
    """
    REQUIREMENT 9a: Validate scores between 0 and 100.
    """
    while True:
        value = input(prompt)
        try:
            score = float(value)
            if 0 <= score <= 100:
                return score
            else:
                print("The input you entered was invalid")
        except ValueError:
            print("The input you entered was invalid")


# =====================================================
# REQUIREMENT 10: round_to_category(score)
# =====================================================

def round_to_category(score):
    """
    REQUIREMENT 10:
    - Accepts a float or int score.
    - Rounds it to the nearest category boundary.
    - If exactly halfway, round UP.
    - Returns (rounded_score, category).
    """

    # Clamp to [0,100]
    if score < 0:
        score = 0
    if score > 100:
        score = 100

    # Define boundaries for categories.
    # You can tweak these to match your scheme.
    # Includes 75 and 85 so examples like 73.7 -> 75, 86.6 -> 85 make sense.
    boundaries = [0, 25, 32, 42, 52, 62, 68, 72, 75, 82, 85, 100]

    nearest = boundaries[0]
    min_diff = abs(score - nearest)

    for b in boundaries[1:]:
        diff = abs(score - b)
        if diff < min_diff:
            min_diff = diff
            nearest = b
        elif diff == min_diff and b > nearest:
            # halfway → round UP
            nearest = b

    rounded_score = int(nearest)
    _, category = determine_category(rounded_score)
    return rounded_score, category


# =====================================================
# REQUIREMENT 12: setup_module()
# =====================================================

def setup_module():
    """
    REQUIREMENT 12:
    - Ask for module name.
    - Ask for number of components.
    - For each: name + weight.
    - Ensure weights sum to 100 (or scale).
    - Return module configuration.
    """
    print("First, let's set up the module configuration.")
    module_name = input("Enter module name: ")

    while True:
        try:
            n = int(input("How many assessment components does this module have? "))
            if n <= 0:
                print("The input you entered was invalid")
                continue
            break
        except ValueError:
            print("The input you entered was invalid")

    component_names = []
    weights = []

    for i in range(n):
        name = input(f"Component {i + 1} name: ")
        component_names.append(name)

        while True:
            try:
                w = float(input(f"Component {i + 1} weight (%): "))
                if w < 0:
                    print("The input you entered was invalid")
                    continue
                weights.append(w)
                break
            except ValueError:
                print("The input you entered was invalid")

    total = sum(weights)
    if abs(total - 100) > 0.0001:
        print("Weights do not sum to 100%. Adjusting proportionally.")
        weights = [w * 100 / total for w in weights]

    print("Module configuration complete.\n")
    return {
        "module_name": module_name,
        "component_names": component_names,
        "weights": weights
    }


# =====================================================
# REQUIREMENT 11: advanced(filename, weights)
# =====================================================

def advanced(filename, weights, component_names):
    """
    REQUIREMENT 11:
    - Read IDs, names, DOBs from filename (StudentData.txt).
    - Ask user for scores for each component.
    - Use all other functions (calculate_overall_score, round_to_category, etc.).
    - Produce a text file output.
    """
    students = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("The input file you entered was invalid or not found.")
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.replace(",", " ").split()]
        if len(parts) < 3:
            print("The input you entered was invalid in file line:", line)
            continue

        sid, name, dob_str = parts[0], parts[1], parts[2]

        if not (sid.isdigit() and len(sid) == 2):
            print("The input you entered was invalid for ID in file line:", line)
            continue

        try:
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        except ValueError:
            print("The input you entered was invalid for DOB in file line:", line)
            continue

        print(f"\nEntering scores for {sid} - {name}")

        scores = []
        for cname, w in zip(component_names, weights):
            prompt = f"{cname} score ({w}%): "
            scores.append(input_score(prompt))

        raw = calculate_overall_score(scores, weights)
        rounded, category = round_to_category(raw)
        age = calculate_age(dob)

        students.append({
            "ID": sid,
            "Name": name,
            "DOB": dob,
            "Age": age,
            "Raw Score": raw,
            "Rounded Score": rounded,
            "Category": category
        })

    students.sort(key=lambda s: s["ID"])

    headers = ["UID", "Name", "D.o.B", "Age", "Raw Score", "Rounded Score", "Category"]
    rows = []
    for s in students:
        rows.append([
            s["ID"],
            s["Name"],
            s["DOB"].isoformat(),
            s["Age"],
            f"{s['Raw Score']:.4f}",
            s["Rounded Score"],
            s["Category"]
        ])

    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    print("\nAdvanced results:\n")
    print(table_str)

    with open("students_advanced.txt", "w", encoding="utf-8") as f:
        f.write(table_str)


# =====================================================
# REQUIREMENTS 1–10 MAIN PROGRAM
# - 1: uses functions
# - 2: input expansion (ID, name, DOB, then scores)
# - 3: loop for up to 3 students or 'end'
# - 4 & 5: store scores and categories
# - 6: store ages
# - 7: tabulate sorted by ID
# - 8: write to students.txt
# - 9: validation helpers above
# - 10: uses round_to_category
# =====================================================

def main():
    print("Welcome to the Student Grading System")

    # REQUIREMENT 12 LINK: use setup_module() if desired
    choice = input("Configure new module? (yes/No): ").lower()
    if choice == "yes":
        module_config = setup_module()
        component_names = module_config["component_names"]
        weights = module_config["weights"]
    else:
        # Coursework 1 default
        component_names = ["Coursework 1", "Coursework 2", "Coursework 3", "Final Exam"]
        weights = [10, 20, 30, 40]

    print("\nNow, let's enter student information and grades.\n")

    students = []  # REQUIREMENTS 4,5,6,9b,c – list of dicts

    # REQUIREMENT 3: loop up to 3 students or 'end'
    while len(students) < 3:
        sid = input_student_id()
        if sid == "end":
            break

        # REQUIREMENT 2: ID → name → DOB → scores
        name = input("Enter student name: ")
        dob = input_dob()
        age = calculate_age(dob)

        scores = []
        for cname in component_names:
            scores.append(input_score(f"{cname} score: "))

        # REQUIREMENT 4: overall score (raw)
        raw_score = calculate_overall_score(scores, weights)

        # REQUIREMENT 10: rounded + category
        rounded_score, category = round_to_category(raw_score)

        # REQUIREMENTS 4,5,6: store for later use
        students.append({
            "ID": sid,
            "Name": name,
            "DOB": dob,
            "Age": age,
            "Raw Score": raw_score,
            "Rounded Score": rounded_score,
            "Category": category
        })

    # REQUIREMENT 7: tabulate sorted by ID
    students.sort(key=lambda s: s["ID"])

    if students:
        headers = ["UID", "Name", "D.o.B", "Age", "Raw Score", "Rounded Score", "Category"]
        rows = []
        for s in students:
            rows.append([
                s["ID"],
                s["Name"],
                s["DOB"].isoformat(),
                s["Age"],
                f"{s['Raw Score']:.4f}",
                s["Rounded Score"],
                s["Category"]
            ])

        table_str = tabulate(rows, headers=headers, tablefmt="grid")
        print("\nResults:\n")
        print(table_str)

        # REQUIREMENT 8: save table to students.txt
        with open("students.txt", "w", encoding="utf-8") as f:
            f.write(table_str)


if __name__ == "__main__":
    main()

    # Example call for REQUIREMENT 11 after everything else:
    # module_config = setup_module()
    # advanced("StudentData.txt", module_config["weights"], module_config["component_names"])
