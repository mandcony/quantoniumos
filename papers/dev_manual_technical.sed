# Remove simplex command definition
/\\newcommand{\\simplex}/d
# Remove hardcore command definition  
/\\newcommand{\\hardcore}/d
# Remove the "How to Read This Manual" section
/\\section\*{How to Read This Manual}/,/^$/d
# Remove simplex blocks (handle multiline)
/\\simplex{/,/}$/d
# Remove \hardcore{ prefix, keep content
s/\\hardcore{//g
# Clean subtitle
s/For Technical and Non-Technical Audiences/Technical Reference/g
# Clean abstract
s/plain-English explanations for newcomers and //g
s/\*\*hardcore technical implementations\*\*/technical implementations/g
