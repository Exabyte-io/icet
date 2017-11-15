Contribution guidelines
=======================

Some general guidelines:

* **Read the contribution guidelines**.<br>
  You have done well so far but keep on reading all the way to the **bottom of
  this page**.

* **Remember and apply the contribution guidelines!**<br>
  [(random user guideline quizzing can occur at any
  time)](http://www.ohsinc.com/services/employee-drug-testing/)

* Use [expressive function and variable names](https://xkcd.com/910/), e.g.,
  * Good: `get_number_of_structures` (Python), `getNumberOfStructures` (C++)
  * Avoid: `get_nbr_struct` or any variation thereof
  * Good: `get_number_of_allowed_elements` (Python),
    `getNumberOfAllowedElements` (C++)
  * Avoid: `get_Mi` or any variation thereof
  * Avoid anything remotely resembling `int mytmp = 13`

* Document and comment, document and comment, document and comment, document
  and comment, document and comment ... (it cannot be said too often).<br>
  Language specific guide lines are given below.

* Always imagine someone else has to be able to read your code. Hence do your
  best at writing [*clean and structured* code](https://www.xkcd.com/1513/).
  * Avoid commenting out blocks of code for "later use/reference" in commits.
  * When writing C++ code, separate declaration and definition in `*.hpp` and
    `*.cpp` files. This is not just a matter of good coding style but
    [compilation time during development](https://xkcd.com/303/).


C++
---

This project adopts a more concise [style](https://www.xkcd.com/1695/) when
writing C++ code that is in the spirit of
[K&R](https://en.wikipedia.org/wiki/Indent_style) and
[pep7](https://www.python.org/dev/peps/pep-0007/) (yes, that's a thing).

Any functions/functionality *must* be properly documented. The API documentation
is generated using [doxygen](http://www.stack.nl/~dimitri/doxygen/). You should
therefore include comment blocks that document your code and are formatted to
comply with the doxygen markup style.

For most functions, class members, etc. that can be comprehensively described
using a single line one can use the triple-slash form, e.g.,
```
private:
   /// Space group number according to ITCA.
   int _spacegroup;

public:
   /// Returns the space group number.
   int getSpacegroup() { return _spacegroup; }
```

For more substantial functions, classes, or other elements (such as ``enum``-s)
adopt the extended documentation form
```
/**
  @brief Write a structure to file.
  @details This function writes an atomic structure to file using different formats.
  @param struct   The atomic configuration
  @param filename The name of the output file.
  @param format   The output file format; possible values: 'vasp', 'xyz'\
*/
void writeStructureToFile(AtomicStructure *struct, string::string filename, string::string format) {
    ...
}
```
Usually, declaration and definition are split between `*.hpp` and `*.cpp` files.
In that case, the following variation is preferred. In the `*.hpp` file:
```
/// Write a structure to file.
void writeStructureToFile(AtomicStructure *, string::string, string::string);
```
In the `*.cpp` file:
```
/**
  @details This function writes an atomic structure to file using different formats.
  @param struct   The atomic configuration
  @param filename The name of the output file.
  @param format   The output file format; possible values: 'vasp', 'xyz'
*/
void writeStructureToFile(AtomicStructure *struct, string::string filename, string::string format) {
    ...
}
```
More examples can of course be found in the code.

Please use [CamelCase](https://en.wikipedia.org/wiki/Camel_case) and [expressive
names](https://xkcd.com/302/) for functions, classes, and members (avoiding
unnecessary and non-standard abbreviations). Good examples are
``writeStructureToFile``, ``AtomicStructure``. Private and
protected class members should be preceded by an underscore as in
``_myPrivateVariable``.

Please ensure [const
correctness](https://isocpp.org/wiki/faq/const-correctness).


Python
------

Code should be [pep8](https://www.python.org/dev/peps/pep-0008/) compliant and
pass [pyflakes](https://pypi.python.org/pypi/pyflakes). (Eventually,
pep8/pyflakes will be added to the CI, at which point code *must* be
compliant.)

Any functions/functionality *must* be properly documented. This includes
[docstrings](https://en.wikipedia.org/wiki/Docstring) for functions, classes,
and modules that clearly describe the task performed, the interface (where
necessary), the output, and if possible an example. atomicrex uses [NumPy Style
Python Docstrings](http://sphinxcontrib-
napoleon.readthedocs.io/en/latest/example_numpy.html).

When in doubt ask the main developers. Also [the coding conventions from
ASE](https://wiki.fysik.dtu.dk/ase/development/python_codingstandard.html)
provide useful guidelines.

Good job, you are still reading! [Will you make it to the
end?](https://xkcd.com/169/)


Please use spaces
-----------------

While you are entitled to [your own
opinion](http://lea.verou.me/2012/01/why-tabs-are-clearly-superior/) this
project uses spaces instead of tabs. Even if you are geeky enough to care and
like [Silicon Valley](https://www.youtube.com/watch?v=SsoOG6ZeyUI) you should
know that [developers who use spaces make more
money](https://stackoverflow.blog/2017/06/15/developers-use-spaces-make-money-use-tabs/).
Also the use of spaces is strongly recommended by our beloved
[pep8](https://www.python.org/dev/peps/pep-0008/) standard.


Commits/issues/merge requests
-----------------------------

### General guidelines

Bug reports, features suggestions, code review etc must be handled via gitlab.
The following [workflow](https://xkcd.com/1172/) is *strongly* encouraged:
* creatie an issue
* invite comments from users, developers, and the [product
  owner](https://en.wikipedia.org/wiki/Scrum_(software_development)#Product_owner)
  via the GitLab interface (e.g., use `@username` to address specific team
  members in issue descriptions, messages, or discussions)
* review the input; in the case of *any* somewhat larger issues the decision
  whether (and how) to move ahead *must* be discussed and approved by at least
  one more experienced team member, preferably the product owner
* create a branch from the issue via the GitLab interface
* once the work on the issue has been completed *first* clean up your code and
  review the [items that will be covered during code review (see
  below)](http://commadot.com/wtf-per-minute/); *then* submit a merge request
* the merge request must be [reviewed by *another* developer](https://www.xkcd.com/1833/) for
  * code passes all existing tests
  * functionality
  * performance
  * code quality
  * compliance to style guide
  * addition of new unit tests
  In this step, the responsibility for making the code compliant resides with
  the developer *not* the reviewer.
* if the code review is successful the code is merged into master by the
  reviewer
For almost all issues, the time from creating a branch to merging into master
should not exceed two weeks (one week is preferable).


### Commit messages

When writing commit messages, generating issues, or submitting merge requests,
[write meaningful and concise commit messages](https://xkcd.com/1296/). Also
please use the following prefixes to identify the category of your
commit/issue/merge request.

* BLD: change related to building
* BUG: bug fix
* DATA: general data
* DOC: documentation
* ENH: enhancement
* MAINT: maintenance commit (refactoring, typos, etc.)
* STY: style fix (whitespace, PEP8)
* TST: addition or modification of tests
* REL: related to releases

Less common:
* API: an (incompatible) API change
* DEP: deprecate something, or remove a deprecated object
* DEV: development tool or utility
* FIG: images and figures
* REV: revert an earlier commit

The first line should not exceed 78 characters. If you require more
space, insert an empty line after the "title" and add a longer message
below. In this message, you should again limit yourself to 78
characters *per* line.

Hint: If you are using emacs you can use ``Meta``+``q``
[shortcut](https://shortcutworld.com/en/Emacs/23.2.1/linux/all) to
"reflow the text". In sublime you can achieve a similar effect by using
``Alt``+``q`` (on Linux)/ ``Alt``+``Cmd``+``q`` (on MacOSX).
