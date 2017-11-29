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

Code should be [pep8](https://www.Python.org/dev/peps/pep-0008/) compliant and
pass [pyflakes](https://pypi.Python.org/pypi/pyflakes). (Eventually,
pep8/pyflakes will be added to the CI, at which point code *must* be
compliant.)

Any functions/functionality *must* be properly documented. This includes
[docstrings](https://en.wikipedia.org/wiki/Docstring) for functions, classes,
and modules that clearly describe the task performed, the interface (where
necessary), the output, and if possible an example. atomicrex uses [NumPy Style
Python Docstrings](http://sphinxcontrib-
napoleon.readthedocs.io/en/latest/example_numpy.html).

When in doubt ask the main developers. Also [the coding conventions from
ASE](https://wiki.fysik.dtu.dk/ase/development/Python_codingstandard.html)
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
[pep8](https://www.Python.org/dev/peps/pep-0008/) standard.


Commits/issues/merge requests
-----------------------------

### General guidelines

Bug reports, features suggestions, code review etc must be handled via gitlab.
The following [workflow](https://xkcd.com/1172/) is *strongly* encouraged:

__Preparation phase:__

As an issue reporter:
* Create an issue in the Backlog. 
   * When creating an issue for a __feature request__, it is good practise to 
     describe who the issue is to be developed for, what should be achieved, 
     and ensure that implementation of the feature result in user value. 
     Good practise is to write a 
     [user story](https://www.mountaingoatsoftware.com/agile/user-stories), 
     e.g.:
         1. _As a user at a neutron source, I would like to be able to compare_ 
            _my acquired powder diffraction data with `icet` simulations, so that_
            _I can explain my temperature dependent observations._
         2. _As a developer, I would like to easily figure out how to best_
            _contribute to the `icet` project._

         For a discussion of good user stories, see 
     [discussion on Mountain Goat's homepage](https://www.mountaingoatsoftware.com/agile/user-stories)
     It is also a good idea to describe how a succesful implementation can be 
     demonstrated. For example for the above user stories we can write:
         1. DEMO: Demonstrate that a user via Python can;
             1. Import neutron powder diffraction data 
             2. Carry out an `icet`/`mchammer` simulation at finite temperature
             3. Calculate the scattering curves from the ensemble obtained from 
                the `icet`/`mchammer` simulation
             4. Plot experimental data and simulation data in the same widget
         2. DEMO: A potential developer, rather than an actual developer, can
            1. find the contribution guidelines when pointed to the `icet` 
               homepage
            2. Contribute successfully to the code (i.e. pass a review) after
               having read the guidelines.
               
   * When creating a __bug report__, it is important to describe how the bug can 
     be reproduced.
* Invite comments from users, developers, and the [product
  owner](https://en.wikipedia.org/wiki/Scrum_(software_development)#Product_owner) 
  and other stakeholders via the GitLab interface (e.g., use `@username` to 
  address specific team members in issue descriptions, messages, or discussions)
* Review the input and make adjustments according to input. If the issue is 
  considered to take more than five days to complete, try to split it into 
  smaller ones, each with their own user story.
* Once the issue is ready for development, the product owner and *only* the 
  product owner (or a person appointed by the product owner) can move the issue 
  to the `To Do` column. The product owner may choose to assign the issue to a 
  developer or let it be up for grasp for anyone interested.

__Development phase:__

As a developer:
* If the issue is not already assigned to you, assign it to yourself. Move the 
  issue to the `Doing` column.
* Create a branch from the issue via the GitLab interface. Ensure that other 
  people can see from the issue which branch you are working on. 
* Once the work on the issue has been completed *first* clean up your code and
  review the [items that will be covered during review (see
  below)](http://commadot.com/wtf-per-minute/). Make sure that the described 
  demonstration can be done, that there are unit tests (or regression tests in
  case of bug fixes), and developer and user documentation. 
* Move the issue to the column `Review`. Reassign the issue to [*another* 
  developer](https://www.xkcd.com/1833/) or leave the issue unassigned and thus 
  up for grasp by *another* developer

__Review phase:__

As a reviewer:
* If the issue up for review is not already assigned to you, assign it to 
  yourself. 

* The development related to the issue must be reviewed for
  * sufficient test coverage. 
  * code passes all existing tests
  * functionality
  * performance
  * code quality
  * compliance to style guide
  * addition of new unit tests
  * demonstration is fulfilled

* If the code review is successful the code is merged into master by the
  reviewer. If the review is unsuccesful, the reviewer's comments are added as a 
  comment to the issue. Reviewer moves the issue back to the `To Do` column and 
  reassigns the issue to the developer. 
  The responsibility for making the code compliant resides with
  the developer *not* the reviewer.
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

The first line should not exceed 78 characters. If you require more
space, insert an empty line after the "title" and add a longer message
below. In this message, you should again limit yourself to 78
characters *per* line.

Hint: If you are using emacs you can use ``Meta``+``q``
[shortcut](https://shortcutworld.com/en/Emacs/23.2.1/linux/all) to
"reflow the text". In sublime you can achieve a similar effect by using
``Alt``+``q`` (on Linux)/ ``Alt``+``Cmd``+``q`` (on MacOSX).
