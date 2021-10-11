# REFINED generator
Generate REFINED 2D image representations for numerical data

## Use as a library

```python
    process = Refined()
    process.fit(data)
    process.plot_mapping()
    process.save_mapping_for_hill_climbing()
    process.generate_image(data, 'Test', 'jpg')
```

## Use as CLI



Note:

the best dim-reduction for LAP is probably TSNE. 
