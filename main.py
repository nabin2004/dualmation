from dualmation.pipeline import DualAnimatePipeline, PipelineConfig

config = PipelineConfig(
    concept="Explain gradient descent visually",
    llm_model="codellama/CodeLlama-7b-hf",
    diffusion_model="stabilityai/stable-diffusion-2-1",
)

pipeline = DualAnimatePipeline(config)
result = pipeline.run()
print(result)