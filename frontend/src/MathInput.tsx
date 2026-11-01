import { useEffect, useRef } from "react";
import { MathfieldElement } from "mathlive";
import "mathlive/fonts.css";
import { Button } from "@/components/ui/button";
MathfieldElement.fontsDirectory = null; // Vite bundles fonts from fonts.css.
MathfieldElement.soundsDirectory = null;
MathfieldElement.computeEngine = null; // Explicit PDE parser owns mathematical meaning.
const symbols = [
  ["∂", String.raw`\partial`],
  ["∇", String.raw`\nabla`],
  ["∇·", String.raw`\nabla\cdot`],
  ["Δ", String.raw`\Delta`],
  ["subscript", String.raw`_{#0}`],
  ["power", String.raw`^{#0}`],
  ["fraction", String.raw`\frac{#0}{#?}`],
  ["( )", String.raw`\left(#0\right)`],
  ["vector", String.raw`\mathbf{#0}`],
  ["x", "x"],
  ["y", "y"],
  ["z", "z"],
  ["τ", String.raw`\tau`],
  ["u", "u"],
  ["a", "a"],
  ["D", "D"],
  ["ν", String.raw`\nu`],
  ["λ", String.raw`\lambda`],
];
export function MathInput({
  value,
  onChange,
  index,
  names = [],
}: {
  value: string;
  onChange: (v: string) => void;
  index: number;
  names?: string[];
}) {
  const host = useRef<HTMLDivElement>(null),
    field = useRef<MathfieldElement | null>(null),
    callback = useRef(onChange);
  useEffect(() => {
    callback.current = onChange;
  }, [onChange]);
  useEffect(() => {
    const element = new MathfieldElement();
    element.mathVirtualKeyboardPolicy = "manual";
    element.setAttribute("aria-label", `Equation ${index + 1}`);
    element.setAttribute("role", "textbox");
    element.setAttribute("tabindex", "0");
    element.style.width = "100%";
    const input = () => callback.current(element.value);
    element.addEventListener("input", input);
    host.current?.appendChild(element);
    field.current = element;
    return () => {
      element.removeEventListener("input", input);
      element.remove();
      field.current = null;
    };
  }, [index]);
  useEffect(() => {
    if (field.current && field.current.value !== value)
      field.current.setValue(value, { silenceNotifications: true });
  }, [value]);
  return (
    <div className="math-input">
      <div
        className="symbol-toolbar"
        role="toolbar"
        aria-label={`Symbols for equation ${index + 1}`}
      >
        {symbols
          .filter(([label]) =>
            ["∂", "∇·", "Δ", "fraction", "subscript", "τ"].includes(label),
          )
          .map(([label, latex]) => (
            <Button
              key={label}
              variant="outline"
              size="sm"
              aria-label={`Insert ${label} in equation ${index + 1}`}
              onClick={() => {
                field.current?.focus();
                field.current?.insert(latex);
                if (field.current) callback.current(field.current.value);
              }}
            >
              {label}
            </Button>
          ))}
        {[...new Set(names)]
          .filter(
            (name) =>
              !["∂", "∇·", "Δ", "fraction", "subscript", "τ"].includes(name),
          )
          .slice(0, 9)
          .map((name) => (
            <Button
              key={name}
              variant="outline"
              size="sm"
              aria-label={`Insert ${name} in equation ${index + 1}`}
              onClick={() => {
                field.current?.focus();
                field.current?.insert(name);
                if (field.current) callback.current(field.current.value);
              }}
            >
              {name}
            </Button>
          ))}
        <details>
          <summary>More symbols</summary>
          <div className="symbol-toolbar">
            {" "}
            {symbols
              .filter(
                ([label]) =>
                  ![
                    "∂",
                    "∇·",
                    "Δ",
                    "fraction",
                    "subscript",
                    "τ",
                    ...names,
                  ].includes(label),
              )
              .map(([label, latex]) => (
                <Button
                  key={label}
                  variant="outline"
                  size="sm"
                  aria-label={`Insert ${label} in equation ${index + 1}`}
                  onClick={() => {
                    field.current?.focus();
                    field.current?.insert(latex);
                    if (field.current) callback.current(field.current.value);
                  }}
                >
                  {label}
                </Button>
              ))}
          </div>
        </details>
      </div>
      <div ref={host} />
    </div>
  );
}
