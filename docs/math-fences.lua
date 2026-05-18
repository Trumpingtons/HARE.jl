-- Convert ```math fenced code blocks to display math for pandoc/LaTeX
function CodeBlock(el)
  if el.classes[1] == "math" then
    return pandoc.RawBlock("latex", "\\[\n" .. el.text .. "\n\\]")
  end
end
