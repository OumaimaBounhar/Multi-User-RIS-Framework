class multiplicativeDecaySchedule:
    """ 

    Apply a decay to a certain value until reaching a minimum value :
        value <- max(value * decay, min_value)

    Works for epsilon and delta.

    """
    def __init__(self, init_value: float, decay: float, min_value: float):
        assert min_value <= init_value
        assert 0.0 < decay <= 1.0
        self.init_value = init_value
        self.value = init_value
        self.decay = decay
        self.min_value = min_value
    
    def step(self) -> float:
        self.value =  max(self.value * self.decay, self.min_value)
        return self.value
    
    def get(self) -> float:
        return self.value
    
    def reset(self) -> None:
        self.value = self.init_value

    def change(self, new_value: float) -> None:
        self.value = new_value

class LinearDecaySchedule:
    """
    Linear decay from init_value to min_value over total_steps.
    value(t) <- max(init_value - (init_value-min_value)*t/total_steps, min_value)
    """
    def __init__(self, init_value: float, min_value: float, total_steps: int):
        assert min_value <= init_value
        assert total_steps > 0
        self.init_value = init_value
        self.value = init_value
        self.min_value = min_value
        self.time_step = 0
        self.total_steps = total_steps

    def step(self) -> float:
        self.time_step += 1
        frac = min(self.time_step / self.total_steps, 1.0)
        self.value = max(self.init_value - (self.init_value - self.min_value) * frac, self.min_value)
        return self.value

    def get(self) -> float:
        return self.value

    def reset(self) -> None:
        self.time_step = 0
        self.value = self.init_value