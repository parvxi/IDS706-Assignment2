# Rust and Ownership Notes (Week 2)


For the Rust part, I worked through:

`notebooks/rust_vs_python_intro.ipynb`

using the `evcxr_jupyter` Rust kernel.

I completed the exercises and also changed some of the examples myself to see what errors Rust would give me.

| Experiment                                         | What happened                         |
| -------------------------------------------------- | ------------------------------------- |
| Reassigning a regular `let` variable               | Rust gave an immutable variable error |
| Adding `mut` and reassigning                       | Worked                                |
| Moving a vector and then using the original        | Rust gave a moved-value error         |
| Removing `.clone()`                                | Produced the ownership error          |
| Trying to change a vector while looping through it | Rust rejected the code                |
| Borrowing with `&` instead of moving or cloning    | Both names worked, nothing was copied |

The ownership part was probably the biggest difference I noticed compared with Python.

In Python, I normally do not think much about who owns a list or whether a variable is allowed to change. Rust makes those rules much more explicit and will not compile the program if they are broken.

The `.clone()` example also helped me understand that copying data is an actual operation with a cost. Rust makes me ask for that copy explicitly instead of doing it without thinking about it.

The notebook shows borrowing with & but does not have an exercise for it, so I added a cell to try it myself. Both names still worked and nothing was copied, which showed me there is a third option besides moving or cloning.

**Note:** I am still new to Rust, but experimenting with the errors helped the ownership rules make more sense to me than only reading about them.
