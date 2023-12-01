import torch._C as C 

def launch():
    C._dao_launch()

def sync():
    C._dao_sync()
    
def verbose(level = 1):
    C._dao_verbose(level)

def status():
    C._dao_status()

def stop():
    C._dao_stop()