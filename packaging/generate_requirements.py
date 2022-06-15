"""Generate a requirements.txt automatically using only modules imported in shocksurvey package."""
import imp
import os
import re
import warnings

print("-" * 80)
MODULE_PATH = "/Users/jamesplank/Documents/PHD/shocks_survey/shocksurvey"

EXCLUDED_FILES = ["setup.py", ".pyc"]
EXCLUDED_MODULES = ["shocksurvey", ""]

# Get all files in directory
files = []
for (dirpath, dirnames, filenames) in os.walk(MODULE_PATH):
    files.extend([os.path.join(dirpath, f) for f in filenames])

print("ALL FILES:", "\n".join(["    " + f for f in files]), sep="\n")
print("-" * 80)
# Filter to only python files that aren't blacklisted
files = [f for f in files if ".py" in f and not any(ex in f for ex in EXCLUDED_FILES)]
print("PYTHON FILES: ", "\n".join(["    " + f for f in files]), sep="\n")
# Extract imported modules from files in shocksurvey
modules = []
for f in files:
    print("-" * 80)
    print(f"READING FILE: {f.split('shocks_survey/')[1]}")
    with open(f, "r") as File:
        file = File.read().splitlines()
    file_modules = []
    for line in file:
        import_line = line.split()

        if line.startswith("from"):
            module = import_line[import_line.index("from") + 1].split(".")[0]
            file_modules.append(module)
        elif line.startswith("import"):
            module = import_line[import_line.index("import") + 1].split(".")[0]
            file_modules.append(module)
    print(" :: ".join(file_modules))
    modules.extend(file_modules)

print("-" * 80)
modules = list(set(modules))  # No repeated modules
modules = [mod for mod in modules if mod not in EXCLUDED_MODULES]
print("ALL MODULES FROM FILES: ", modules)

# Remove all standard library modules
third_party_modules = []
for mod in modules:
    mod_path = imp.find_module(mod)[1]
    # pip installs to virtualenv called shockstat
    if ".virtualenvs/shockstat" in mod_path:
        third_party_modules.append(mod)
modules = third_party_modules

print("ALL THIRD PARTY MODULES: ", modules)
print("-" * 80)
# Extract pip installed modules
pip_modules = os.popen("pip freeze").read().splitlines()
# Get pip name for each module
required_modules = []
for mod in modules:
    match_found = False
    for p in pip_modules:
        # Regex finds common variations on module name, e.g. MODULE, pyMODULE, MODULE-python
        match = re.findall(
            f"((?:py(?:thon)?-?)?{mod}(?:-?py(?:thon)?)?(?===).*)", p, re.I
        )
        if len(match) > 0:
            required_modules.append(match[0].replace("==", ">="))
            match_found = True
            break
    if not match_found:
        # Some modules are called something stupid so warn if match cannot be found
        warnings.warn(f"No match found for module {mod} in pip.")

print("REQUIRED PIP MODULES: ", required_modules)
print("-" * 80)
print("WRITING FILE TO: ", os.path.join(MODULE_PATH, "requirements.txt"))
with open(os.path.join(MODULE_PATH, "requirements.txt"), "w") as file:
    file.write("\n".join(required_modules))
print("WRITTEN FILE")
print("-" * 80)
