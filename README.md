# ina-segmenter-modified

Credit To: https://github.com/ina-foss/inaSpeechSegmenter

## Important Modification

We added numerical outputs for probabilities of each gender 

Modified Usage: 

```py
seg = Segmenter()
data = seg(file_path)
df = pandas.DataFrame(data, columns=['Prediction', 'Start', 'End', 'Confidence'])
print(df)
```

Output Format: `list[ResultFrame(prediction, start_time_seconds, end_time_seconds, confidence or probability)]`

