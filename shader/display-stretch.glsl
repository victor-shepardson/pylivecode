in vec2 uv;

uniform vec2 size;
uniform sampler2D color;

out vec4 fragColor;

void main() {
  // vec2 px = gl_FragCoord.xy;
  // vec2 s = textureSize(color, 0);
  // vec2 p = px/s;

  fragColor = texture(color, uv);
}
