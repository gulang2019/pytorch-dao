We need to maintain a table to record each tensors' lifetime in logical time axis.

Generator:
```pseudo-code
Init t = 0; 
push_kernel(inputs, output, func);
t += 1;
for input in inputs:
    if (LastAccess(input)) 
        tensor_table.set_end_time(input, t); 
```

Executor
```pseudo code

```