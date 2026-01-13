import math

class multiplicativeDecaySchedule:
    """ 

    Apply a decay to a certain value until reaching a minimum value :
        value <- max(value * decay, min_value)

    Works for epsilon and delta.

    """
    def __init__(self, init_value, decay, min_value):
        self.init_value = init_value
        self.value = init_value
        self.decay = decay
        self.min_value = min_value
    
    def step(self):
        self.value =  max(self.value * self.decay, self.min_value)
        return self.value
    
    def get(self):
        return self.value
    
    def reset(self):
        self.value = self.init_value

    def change(self, new_value):
        self.value = new_value
        