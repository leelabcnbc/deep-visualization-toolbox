# Deep Visualization Toolbox

## Caveats

1. Use OpenCV 2, not 3.
2. in your `deploy.prototxt`, have `force_backward: true` and make sure the batch size is 1. Otherwise, a lot of things would break.
