
AIRM_PREFIX = "Intel® AI Reference Models -"
ROOT_STR_LEN = "./".__len__()
REQUIREMENTS_STR_LEN = "/requirements.txt".__len__()
DEBUG=False

def debug (msg):
  if DEBUG:
    print (f"{msg}")

def error (msg):
  print (f"ERROR! {msg}")
  exit(1)
