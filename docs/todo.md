
# Data Layer Refactoring.
- [ ] We will be using TDD. Modular and testable code.
- [ ] Public interface will be DataManager. It will provide everything from providers.
  - [ ] FileProvider for local files.
  - [ ] ApiProvider for remote APIs.
  - [ ] LiveProvider with websocket support.
- We will design the DataManager like it is live data for both live and file data.
