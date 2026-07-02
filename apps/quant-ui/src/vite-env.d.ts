/// <reference types="vite/client" />

declare module "*.css";

declare global {
  namespace JSX {
    type Element = import("react").JSX.Element;
  }
}

export {};
