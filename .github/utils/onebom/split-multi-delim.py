import re
import sys

requirement_line = sys.argv[1].strip()
#print f("requirement_line={}", requirement_line)
if not requirement_line.startswith("#") :
  requirement = re.split(r'\s|~|<|>|=|;|\[', requirement_line)[0]
  if requirement is not None and requirement != "":
    requirement = requirement.replace("_", "-")
    print (requirement)
  else:
    print ("")
  sys.exit(0)
sys.exit(-1)
