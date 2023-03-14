from zeroshot import ZeroShotPipeline


pipeline = ZeroShotPipeline()

predictions = pipeline.predict('pup.jpg',['dog','person'])

for label,prob in predictions:
    print(f"{label}: {prob:.2f}")