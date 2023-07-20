from metaflow import metaflow, FlowSpec, step, Flow

class TestFlow(FlowSpec):
    
    @step
    def start(self):
        run = Flow('')

flow = list(Metaflow())[1]
run = flow.latest_run
