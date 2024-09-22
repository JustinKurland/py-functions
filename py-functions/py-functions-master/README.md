[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://git.nmlv.nml.com/JET5252/py-functions">
    <img src="images/logo.png" alt="Logo" width="300" height="200">
  </a>

  <h3 align="center">Python Functions</h3>

  <p align="center">
    Python functions for various data science tasks meant to reduce the need/dependency for Northwestern Mutual data scientists/engineers to continually repeat tasks or develop bespoke functions for things that myself and other data scientists/engineers may have already done.
    <br />
    <a href="https://git.nmlv.nml.com/JET5252/py-functions"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://git.nmlv.nml.com/JET5252/py-functions">View Demo</a>
    ·
    <a href="https://git.nmlv.nml.com/JET5252/py-functions/-/issues">Report Bug</a>
    ·
    <a href="https://git.nmlv.nml.com/JET5252/py-functions/-/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

There are many great data scientists across Northwestern Mutual, however, I am unaware of any current repository for us to actively share functions that will increase our collective productivity by reducing the need to continuously _reinvent the wheel_. Put differently, there are so many general tasks that we collectively engage in, repeatidly, and spending our time working on these things reduces our ability to focus on the more specific aspects of our work. This repository is meant to be a one-stop shop for data scientists across Northwestern Mutual to make contributions for such tasks and to also leverage the contributions of others who commit new work in our collective effort to help each other. The repository will include functions that assist in all areas of data science, including but not limited to: 

* Feature Engineering
* Unsupervised Feature Selection
* Supervised Feature Selection
* Data wrangling
* Visualization
* Modeling 

This repository is important, here's why:

* Our time should be focused on creating amazing models that assist Northwestern Mutual. Work that solves a problem and helps us!
* We should not be doing the same tasks over and over like creating writing functions to create date and time features from scratch.
* We should implement DRY principles, whenever possible. :smiley:

Of course, no one function will serve all our respective work since needs will vary. So I will be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. All data scientists at Northwestern Mutual who would like to or plan to contribue to this repository, thank you, your contribution will help your fellow data scientists!

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Current Functions Built With

* [Pandas](https://pypi.org/project/pandas/)
* [Numpy](https://pypi.org/project/numpy/)
* [Pandas Flavor](https://pypi.org/project/pandas-flavor/)
* [DateTime](https://pypi.org/project/DateTime/)
* [Fast-ML](https://pypi.org/project/fast-ml/) 



<!-- GETTING STARTED -->
## Getting Started

To make best use of any (or all) of the functions in the repository it is strongly adviced that you (1) clone the repo, (2) have the requisite dependencies/libraries and (3) to method chain pandas specific functions have `pandas-flavor`, which is a (simple) API for registering accessors with Pandas objects that will enable you to make best use of the pandas-specific functions for your own work. Pandas-flavor extends Pandas' extension API by:

1. adding support for registering methods as well.
2. making each of these functions backwards compatible with older versions of Pandas.

To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://git.nmlv.nml.com/JET5252/py-functions.git
   ```
2. Install Pandas, Numpy, Pandas-Flavor, DateTime, Fast-ML
   ```sh
   pip install pandas
   pip install numpy
   pip install pandas-flavor
   pip install DateTime
   pip install fast-ml
   ```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://git.nmlv.nml.com/JET5252/py-functions/-/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the Northwestern DSA community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated** and while I am the maintainer of this __gitlab repository__ and hope to make a repository that will host various useful functions for all of DSA I would be delighted to have others from across Northernwestern Mutual contribute. Here are some simple ways to contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b function/AmazingFunction`)
3. Commit your Changes (`git commit -m 'Add some AmazingFunction'`)
4. Push to the Branch (`git push origin feature/AmazingFunction`)
5. Open a Pull Request

<!-- UPDATES -->
## Updates

As additional functions are contributed I will add them and inform all Northwestern Mutual Data Scientists via the relevant DSA Slack Channels (#dsa-datascience and #python). 


<!-- CONTACT -->
## Contact

Please do not hesitate to contact me via slack directly if you would like to contribute, but would like to run something by me first prior to any PR you might be considering.  

Justin Kurland - [Digital Commons](https://people.nml.com/person.aspx?accountname=nm%5Cjet5252) - justinkurland@northwestern.com

Project Link: https://git.nmlv.nml.com/JET5252/py-functions](https://git.nmlv.nml.com/JET5252/py-functions)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/CONTRIBUTERS-1-green?style=for-the-badge
[contributors-url]: https://git.nmlv.nml.com/JET5252/py-functions/-/graphs/master
[forks-shield]: https://img.shields.io/badge/FORKS-0-blue?style=for-the-badge
[forks-url]: https://git.nmlv.nml.com/JET5252/py-functions/-/forks/new
[stars-shield]: https://img.shields.io/badge/STARS-1-orange?style=for-the-badge
[stars-url]: https://git.nmlv.nml.com/JET5252/py-functions/-/starrers
[issues-shield]: https://img.shields.io/badge/ISSUES-0-yellowgreen?style=for-the-badge
[issues-url]: https://git.nmlv.nml.com/JET5252/py-functions/-/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/northwestern-mutual/
[product-screenshot]: images/screenshot.png
