/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/_app";
exports.ids = ["pages/_app"];
exports.modules = {

/***/ "./pages/_app.js":
/*!***********************!*\
  !*** ./pages/_app.js ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ App)\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"react/jsx-dev-runtime\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/link */ \"./node_modules/next/link.js\");\n/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ \"./node_modules/next/router.js\");\n/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styles/globals.css */ \"./styles/globals.css\");\n/* harmony import */ var _styles_globals_css__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(_styles_globals_css__WEBPACK_IMPORTED_MODULE_3__);\n\n\n\n\nfunction Navigation() {\n    const router = (0,next_router__WEBPACK_IMPORTED_MODULE_2__.useRouter)();\n    const navItems = [\n        {\n            href: \"/\",\n            label: \"Dashboard\"\n        },\n        {\n            href: \"/return\",\n            label: \"Return\"\n        },\n        {\n            href: \"/direction\",\n            label: \"Direction\"\n        },\n        {\n            href: \"/volatility\",\n            label: \"Volatility\"\n        },\n        {\n            href: \"/forecast\",\n            label: \"Forecast\"\n        },\n        {\n            href: \"/regime\",\n            label: \"Regime\"\n        }\n    ];\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"header\", {\n        className: \"header\",\n        children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n            className: \"container\",\n            children: [\n                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"h1\", {\n                    style: {\n                        marginBottom: \"1rem\",\n                        fontSize: \"1.5rem\",\n                        fontWeight: \"bold\"\n                    },\n                    children: \"MLOps Finance Dashboard\"\n                }, void 0, false, {\n                    fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                    lineNumber: 20,\n                    columnNumber: 9\n                }, this),\n                /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"nav\", {\n                    className: \"nav\",\n                    children: navItems.map((item)=>/*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)((next_link__WEBPACK_IMPORTED_MODULE_1___default()), {\n                            href: item.href,\n                            className: `nav-link ${router.pathname === item.href ? \"active\" : \"\"}`,\n                            children: item.label\n                        }, item.href, false, {\n                            fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                            lineNumber: 25,\n                            columnNumber: 13\n                        }, this))\n                }, void 0, false, {\n                    fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                    lineNumber: 23,\n                    columnNumber: 9\n                }, this)\n            ]\n        }, void 0, true, {\n            fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n            lineNumber: 19,\n            columnNumber: 7\n        }, this)\n    }, void 0, false, {\n        fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n        lineNumber: 18,\n        columnNumber: 5\n    }, this);\n}\nfunction App({ Component, pageProps }) {\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment, {\n        children: [\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(Navigation, {}, void 0, false, {\n                fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                lineNumber: 42,\n                columnNumber: 7\n            }, this),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"main\", {\n                className: \"container\",\n                children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(Component, {\n                    ...pageProps\n                }, void 0, false, {\n                    fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                    lineNumber: 44,\n                    columnNumber: 9\n                }, this)\n            }, void 0, false, {\n                fileName: \"C:\\\\Users\\\\Dell\\\\Desktop\\\\MLOPS-finance\\\\frontend\\\\pages\\\\_app.js\",\n                lineNumber: 43,\n                columnNumber: 7\n            }, this)\n        ]\n    }, void 0, true);\n}\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9wYWdlcy9fYXBwLmpzIiwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7QUFBNkI7QUFDVztBQUNUO0FBRS9CLFNBQVNFO0lBQ1AsTUFBTUMsU0FBU0Ysc0RBQVNBO0lBRXhCLE1BQU1HLFdBQVc7UUFDZjtZQUFFQyxNQUFNO1lBQUtDLE9BQU87UUFBWTtRQUNoQztZQUFFRCxNQUFNO1lBQVdDLE9BQU87UUFBUztRQUNuQztZQUFFRCxNQUFNO1lBQWNDLE9BQU87UUFBWTtRQUN6QztZQUFFRCxNQUFNO1lBQWVDLE9BQU87UUFBYTtRQUMzQztZQUFFRCxNQUFNO1lBQWFDLE9BQU87UUFBVztRQUN2QztZQUFFRCxNQUFNO1lBQVdDLE9BQU87UUFBUztLQUNwQztJQUVELHFCQUNFLDhEQUFDQztRQUFPQyxXQUFVO2tCQUNoQiw0RUFBQ0M7WUFBSUQsV0FBVTs7OEJBQ2IsOERBQUNFO29CQUFHQyxPQUFPO3dCQUFFQyxjQUFjO3dCQUFRQyxVQUFVO3dCQUFVQyxZQUFZO29CQUFPOzhCQUFHOzs7Ozs7OEJBRzdFLDhEQUFDQztvQkFBSVAsV0FBVTs4QkFDWkosU0FBU1ksR0FBRyxDQUFDQyxDQUFBQSxxQkFDWiw4REFBQ2pCLGtEQUFJQTs0QkFFSEssTUFBTVksS0FBS1osSUFBSTs0QkFDZkcsV0FBVyxDQUFDLFNBQVMsRUFBRUwsT0FBT2UsUUFBUSxLQUFLRCxLQUFLWixJQUFJLEdBQUcsV0FBVyxHQUFHLENBQUM7c0NBRXJFWSxLQUFLWCxLQUFLOzJCQUpOVyxLQUFLWixJQUFJOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFXNUI7QUFFZSxTQUFTYyxJQUFJLEVBQUVDLFNBQVMsRUFBRUMsU0FBUyxFQUFFO0lBQ2xELHFCQUNFOzswQkFDRSw4REFBQ25COzs7OzswQkFDRCw4REFBQ29CO2dCQUFLZCxXQUFVOzBCQUNkLDRFQUFDWTtvQkFBVyxHQUFHQyxTQUFTOzs7Ozs7Ozs7Ozs7O0FBSWhDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vbWxvcHMtZmluYW5jZS1mcm9udGVuZC8uL3BhZ2VzL19hcHAuanM/ZTBhZCJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgTGluayBmcm9tICduZXh0L2xpbmsnO1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcbmltcG9ydCAnLi4vc3R5bGVzL2dsb2JhbHMuY3NzJztcclxuXHJcbmZ1bmN0aW9uIE5hdmlnYXRpb24oKSB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgXHJcbiAgY29uc3QgbmF2SXRlbXMgPSBbXHJcbiAgICB7IGhyZWY6ICcvJywgbGFiZWw6ICdEYXNoYm9hcmQnIH0sXHJcbiAgICB7IGhyZWY6ICcvcmV0dXJuJywgbGFiZWw6ICdSZXR1cm4nIH0sXHJcbiAgICB7IGhyZWY6ICcvZGlyZWN0aW9uJywgbGFiZWw6ICdEaXJlY3Rpb24nIH0sXHJcbiAgICB7IGhyZWY6ICcvdm9sYXRpbGl0eScsIGxhYmVsOiAnVm9sYXRpbGl0eScgfSxcclxuICAgIHsgaHJlZjogJy9mb3JlY2FzdCcsIGxhYmVsOiAnRm9yZWNhc3QnIH0sXHJcbiAgICB7IGhyZWY6ICcvcmVnaW1lJywgbGFiZWw6ICdSZWdpbWUnIH1cclxuICBdO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPGhlYWRlciBjbGFzc05hbWU9XCJoZWFkZXJcIj5cclxuICAgICAgPGRpdiBjbGFzc05hbWU9XCJjb250YWluZXJcIj5cclxuICAgICAgICA8aDEgc3R5bGU9e3sgbWFyZ2luQm90dG9tOiAnMXJlbScsIGZvbnRTaXplOiAnMS41cmVtJywgZm9udFdlaWdodDogJ2JvbGQnIH19PlxyXG4gICAgICAgICAgTUxPcHMgRmluYW5jZSBEYXNoYm9hcmRcclxuICAgICAgICA8L2gxPlxyXG4gICAgICAgIDxuYXYgY2xhc3NOYW1lPVwibmF2XCI+XHJcbiAgICAgICAgICB7bmF2SXRlbXMubWFwKGl0ZW0gPT4gKFxyXG4gICAgICAgICAgICA8TGlua1xyXG4gICAgICAgICAgICAgIGtleT17aXRlbS5ocmVmfVxyXG4gICAgICAgICAgICAgIGhyZWY9e2l0ZW0uaHJlZn1cclxuICAgICAgICAgICAgICBjbGFzc05hbWU9e2BuYXYtbGluayAke3JvdXRlci5wYXRobmFtZSA9PT0gaXRlbS5ocmVmID8gJ2FjdGl2ZScgOiAnJ31gfVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAge2l0ZW0ubGFiZWx9XHJcbiAgICAgICAgICAgIDwvTGluaz5cclxuICAgICAgICAgICkpfVxyXG4gICAgICAgIDwvbmF2PlxyXG4gICAgICA8L2Rpdj5cclxuICAgIDwvaGVhZGVyPlxyXG4gICk7XHJcbn1cclxuXHJcbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIEFwcCh7IENvbXBvbmVudCwgcGFnZVByb3BzIH0pIHtcclxuICByZXR1cm4gKFxyXG4gICAgPD5cclxuICAgICAgPE5hdmlnYXRpb24gLz5cclxuICAgICAgPG1haW4gY2xhc3NOYW1lPVwiY29udGFpbmVyXCI+XHJcbiAgICAgICAgPENvbXBvbmVudCB7Li4ucGFnZVByb3BzfSAvPlxyXG4gICAgICA8L21haW4+XHJcbiAgICA8Lz5cclxuICApO1xyXG59XHJcbiJdLCJuYW1lcyI6WyJMaW5rIiwidXNlUm91dGVyIiwiTmF2aWdhdGlvbiIsInJvdXRlciIsIm5hdkl0ZW1zIiwiaHJlZiIsImxhYmVsIiwiaGVhZGVyIiwiY2xhc3NOYW1lIiwiZGl2IiwiaDEiLCJzdHlsZSIsIm1hcmdpbkJvdHRvbSIsImZvbnRTaXplIiwiZm9udFdlaWdodCIsIm5hdiIsIm1hcCIsIml0ZW0iLCJwYXRobmFtZSIsIkFwcCIsIkNvbXBvbmVudCIsInBhZ2VQcm9wcyIsIm1haW4iXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///./pages/_app.js\n");

/***/ }),

/***/ "./styles/globals.css":
/*!****************************!*\
  !*** ./styles/globals.css ***!
  \****************************/
/***/ (() => {



/***/ }),

/***/ "next/dist/compiled/next-server/pages.runtime.dev.js":
/*!**********************************************************************!*\
  !*** external "next/dist/compiled/next-server/pages.runtime.dev.js" ***!
  \**********************************************************************/
/***/ ((module) => {

"use strict";
module.exports = require("next/dist/compiled/next-server/pages.runtime.dev.js");

/***/ }),

/***/ "react":
/*!************************!*\
  !*** external "react" ***!
  \************************/
/***/ ((module) => {

"use strict";
module.exports = require("react");

/***/ }),

/***/ "react-dom":
/*!****************************!*\
  !*** external "react-dom" ***!
  \****************************/
/***/ ((module) => {

"use strict";
module.exports = require("react-dom");

/***/ }),

/***/ "react/jsx-dev-runtime":
/*!****************************************!*\
  !*** external "react/jsx-dev-runtime" ***!
  \****************************************/
/***/ ((module) => {

"use strict";
module.exports = require("react/jsx-dev-runtime");

/***/ }),

/***/ "react/jsx-runtime":
/*!************************************!*\
  !*** external "react/jsx-runtime" ***!
  \************************************/
/***/ ((module) => {

"use strict";
module.exports = require("react/jsx-runtime");

/***/ }),

/***/ "fs":
/*!*********************!*\
  !*** external "fs" ***!
  \*********************/
/***/ ((module) => {

"use strict";
module.exports = require("fs");

/***/ }),

/***/ "stream":
/*!*************************!*\
  !*** external "stream" ***!
  \*************************/
/***/ ((module) => {

"use strict";
module.exports = require("stream");

/***/ }),

/***/ "zlib":
/*!***********************!*\
  !*** external "zlib" ***!
  \***********************/
/***/ ((module) => {

"use strict";
module.exports = require("zlib");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = __webpack_require__.X(0, ["vendor-chunks/next","vendor-chunks/@swc"], () => (__webpack_exec__("./pages/_app.js")));
module.exports = __webpack_exports__;

})();